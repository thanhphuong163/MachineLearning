import sys, os
print(os.getcwd())
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la
from models.k_mean import k_mean
from models.em import em


def genGauss(
        n=100, 
        loc=np.array([0, 0]), 
        scale=np.array([1, 1])):
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

def plot_comparison(*args):
    lst_X = [X for X in args]
    fig, axs = plt.subplots(nrows=np.round(len(lst_X)/2).astype(int), ncols=2, figsize=(10, 5*np.round(len(lst_X)/2).astype(int)))
    for i, X in enumerate(lst_X):
        labels = np.unique(X[:,-1])
        for label in labels:
            data = X[X[:,-1] == label]
            axs[int(i/2),i%2].scatter(data[:,0], data[:,1], label=f'Cluster {int(label)}')
        axs[int(i/2), i % 2].legend()
    plt.show()

def run_models():
    n = 300
    p = np.array([.5, .3, .2])
    loc = np.array([[-2, -1], [2, 5], [3, -2]])
    scale = np.array([[2, 2], [1, 1], [1, 3]])
    X = genGaussMix(n=n, p=p, loc=loc, scale=scale)
    # # plotGaussMix(X)
    kmean = k_mean.Kmean(num_clusters=3, num_iters=4)
    X_kmean = kmean(X[:, :-1])
    em_model = em.EM(num_clusters=3, num_iters=20, verbose=False)
    X_em = em_model(X[:, :-1])
    plot_comparison(X, X_kmean, X_em)

if __name__ == "__main__":
    run_models()
