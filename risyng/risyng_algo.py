import kernel
import laplacian as lp
import numpy as np


def RISynG_v1(X, b):
    y, cor_mat = kernel.f_corr(X)
    e, lap = lp.f_lap(X)

    #     cov_mat=(cov_mat-cov_mat.mean())/cov_mat.ptp()

    I = np.identity(cor_mat.shape[0])
    N = cor_mat.shape[0]
    one_n = np.ones((N, N)) / N
    G = (1 - b) * (I - cor_mat) + b * lap
    s, u = np.linalg.eigh(G)
    return s, u, G


def RISynG_v2(X, b):
    y, kern = kernel.f_kernel_corr(X)
    e, lap = lp.f_lap(X)
    I = np.identity(kern.shape[0])
    N = kern.shape[0]

    G = (1 - b) * (I - kern) + b * lap

    s, u = np.linalg.eigh(G)
    return s, u, G, kernel


def RISynG_v3(X, b):
    y, kern = kernel.f_kernel_corr(X)
    e, lap = lp.f_lap(X)
    I = np.identity(kern.shape[0])
    N = kern.shape[0]

    G = (1 - b) * (I - kernel / y) + b * lap
    s, u = np.linalg.eigh(G)
    return s, u, G, kernel


def RISynG_v4(X, b):
    X = X.T
    y, kern = kernel.f_kernel_mahal(X)
    e, lap = lp.f_lap(X)
    I = np.identity(kern.shape[0])
    N = kern.shape[0]

    G = (1 - b) * (I - kernel) + b * lap
    s, u = np.linalg.eigh(G)
    return s, u, G, kernel


# def RISynG_v5(X,b):
#     y,kern=kernel.f_kernel_corr(X)
#     e, lap=lp.f_lap(X)
#     I=np.identity(kern.shape[0])
#     N=kern.shape[0]

#     sk,uk=np.linalg.eigh(kernel)
#     sl,ul=np.linalg.eigh(lap)

#     return s,u,G,kernel
