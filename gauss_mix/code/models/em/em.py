import numpy as np
from utils.functions import gauss_func


class EM(object):
    def __init__(self, num_clusters=2, num_iters=10):
        self.num_clusters = num_clusters
        self.num_iters = num_iters

    @staticmethod
    def gamma_z_nk(x_n, k, pi, mean, cov):
        '''
        Posterior probability
        '''
        probs = np.array([pi_j * gauss_func(x_n,mean_j,cov_j) for pi_j,mean_j,cov_j in zip(pi,mean,cov)])
        return probs[k] / np.sum(probs)

    def __call__(self, X:np.array):
        pass

    def test(self, x_n, k, pi, mean, cov):
        return self.gamma_z_nk(x_n, k, pi, mean, cov)
