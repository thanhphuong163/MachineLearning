import numpy as np

class Kmean(object):
    def __init__(self, num_clusters=2, num_iters=10):
        self.num_clusters = num_clusters
        self.num_iters = num_iters

    @staticmethod
    def calc_distance(x1, x2):
        '''
        Calculate Euclide's distance between two points
        '''
        return np.sqrt(np.sum((x1-x2)**2))

    @staticmethod
    def assign_cluster(X:np.array, clusters:np.array):
        '''
        The point x will be assigned to the closest cluster
        '''
        for i, x in enumerate(X):
            X[i,-1] = np.argmax(np.sum((x[:-1].reshape((1, x.shape[0]-1)) - clusters)**2, axis=1)) + 1
        return X
    
    @staticmethod
    def update_cluster(X):
        labels = np.unique(X[:, -1])
        means = list()
        for label in labels:
            data = X[X[:, -1] == label]
            means.append(data[:, :-1].mean(axis=0))
        return np.array(means)

    @staticmethod
    def loss(X, means):
        cost = 0
        for i, mean in enumerate(means):
            data = X[X[:, -1] == i+1][:, :-1]
            cost += np.sum((data-mean)**2)
        return np.sqrt(cost) / X.shape[0]

    def __call__(self, X:np.array):
        means = X[np.random.choice(X.shape[0], self.num_clusters, replace=False), :]
        self.lst_loss = list()
        labels = np.zeros((X.shape[0], 1))
        X = np.concatenate((X, labels), axis=1)
        for i in range(self.num_iters):
            X = self.assign_cluster(X, means)
            self.lst_loss.append(self.loss(X, means))
            means = self.update_cluster(X)
        return X
