import numpy as np
from utils.functions import gauss_func


class EM(object):
    def __init__(self, num_clusters=2, num_iters=10, verbose=False):
        self.num_clusters = num_clusters
        self.num_iters = num_iters
        self.verbose = verbose

    @staticmethod
    def gamma_z_nk(x_n, k, pi, mean, cov):
        '''
        Responsibility: posterior probability that x_n is of the k^th component
        '''
        probs = np.array([pi_j * gauss_func(x_n,mean_j,cov_j) for pi_j,mean_j,cov_j in zip(pi,mean,cov)])
        return probs[k] / np.sum(probs)

    @staticmethod
    def calc_responsibities(X:np.array, pis:np.array, means:np.array, covs:np.array):
        '''
        gamma(z_nk): eq. (9.16), page 435, Pattern Recognitions and Machine Learning
        '''
        Z = np.zeros((X.shape[0], means.shape[0]))
        for n, x_n in enumerate(X):
            probs = np.array([pi_j * gauss_func(x_n, mean_j, cov_j)
                            for pi_j, mean_j, cov_j in zip(pis, means, covs)])
            Z[n,:] = probs.reshape((-1,)) / np.sum(probs)
        return Z

    @staticmethod
    def update_mean(X:np.array, Z:np.array):
        N_k = Z.sum(axis=0).reshape((-1,1))
        return (Z.T @ X) / N_k

    @staticmethod
    def update_cov(X:np.array, Z:np.array, means:np.array):
        N,D = X.shape
        K = means.shape[0]
        covs = np.zeros((K, D, D))
        for k in range(K):
            for n in range(N):
                tmp = (X[n,:] - means[k,:])
                covs[k, :, :] += Z[n, k] * \
                    np.outer(tmp, tmp)
            covs[k,:,:] /= Z.sum(axis=0)[k]
        return covs
    
    @staticmethod
    def update_pi(Z:np.array):
        N = Z.shape[0]
        N_k = Z.sum(axis=0).reshape((-1, 1))
        return N_k / N

    @staticmethod
    def calc_log_likelihood(X:np.array, pis:np.array, means:np.array, covs:np.array):
        log_likelihood = 0
        for x_n in X:
            log_likelihood += np.log(np.sum([pi_j * gauss_func(x_n, mean_j, cov_j)
                                        for pi_j, mean_j, cov_j in zip(pis, means, covs)]))
        return log_likelihood

    def __call__(self, X:np.array):
        # Initialize
        means = X[np.random.choice(X.shape[0], self.num_clusters, replace=False), :]
        covs = np.array([np.diag([1,1]) for i in range(self.num_clusters)])
        pis = np.array([1/self.num_clusters for i in range(self.num_clusters)])
        labels = None
        self.hist_log_likelihood = [self.calc_log_likelihood(X, pis, means, covs)]
        # EM algorithm
        for i in range(self.num_iters):
            # E step
            Z = self.calc_responsibities(X, pis, means, covs)
            # M step
            means = self.update_mean(X, Z)
            covs = self.update_cov(X, Z, means)
            pis = self.update_pi(Z)
            # Assign cluster
            labels = Z.argmax(axis=1) + 1
            # Evaluate Log Likelihood
            log_likelihood = self.calc_log_likelihood(X, pis, means, covs)
            if self.verbose:
                print(f"Epoch #{i+1}: Log likelihood = {log_likelihood}")
            self.hist_log_likelihood.append(log_likelihood)
        if self.verbose:
            print("Means:\n", means)
            print("Covs:\n", covs)
            print("Pis:\n", pis)
        return np.concatenate([X, labels.reshape((-1,1))], axis=1)

    def test(self, X, pis, means, covs):
        Z = self.calc_likelihood(X, pis, means, covs)
        N_k = np.sum(Z, axis=0).reshape(Z.shape[1],1)
        return self.update_cov(X, Z, means)
