import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.k_mean import k_mean


def genGauss(n=100, loc=np.array([0, 0]), scale=np.array([1, 1])):
    dim = loc.shape[0]
    X = np.random.normal(loc=loc, scale=scale, size=(n, dim))
    return X


def genGaussMix(n=100, p=np.array([.5, .5]), loc=np.array([[-2, 0], [2, 0]]), scale=np.array([[1, 1], [1, 1]])):
    k = loc.shape[0]
    lstX = []
    for i in range(k):
        samples = int(round(p[i]*n))
        X = genGauss(n=samples, loc=loc[i, :], scale=scale[i, :])
        Y = np.empty((samples, 1))
        Y.fill(i+1)
        lstX.append(np.concatenate((X, Y), axis=1))
    return np.concatenate(lstX, axis=0)


def plotGaussMix(X):
    labels = np.unique(X[:, -1])
    plt.figure(figsize=(10, 7))
    for i in labels:
        data = X[X[:,-1] == i]
        plt.scatter(data[:, 0], data[:, 1], label=f'Cluster {int(i)}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    n = 300
    p = np.array([.5, .3, .2])
    loc = np.array([[-2, -1], [2, 5], [3, -2]])
    scale = np.array([[2, 2], [1, 1], [1, 3]])
    X = genGaussMix(n=n, p=p, loc=loc, scale=scale)
    # plotGaussMix(X)
    kmean = k_mean.Kmean(num_clusters=3, num_iters=4)
    data, loss = kmean(X[:,:-1])
    # print(data)
    print(loss)
    plotGaussMix(data)
