import numpy as np
import numpy.linalg as la


def gauss_func(x:np.array, mean:np.array, cov:np.array):
    assert x.shape == mean.shape
    assert cov.shape == (x.shape[0], x.shape[0])
    D = x.shape[0]
    pdf = (2*np.pi)**(-D/2) * la.det(cov)**(-0.5) * \
        np.exp(-0.5 * ((x-mean).T @ la.inv(cov) @ (x-mean)))
    return pdf.item()


def z_nk(x_n, pi_k, mean_k, cov_k):
    '''
    Probability of the n^th instance belong to cluster k^th
    '''
    return pi_k * gauss_func(x_n, mean_k, cov_k)

def my_square(n):
    return n*n

def my_pow(n,e):
    ans = 1
    for i in range(e):
        ans *= n
    return ans