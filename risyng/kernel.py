import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


def mahal_sqf(X):
    K = np.zeros((X.shape[0], X.shape[0]))
    cov = np.dot(X.T, X)
    inv = np.linalg.inv(cov)
    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            delta = np.abs(X[i] - X[j])
            K[i][j] = K[j][i] = np.dot(np.dot(delta, inv), delta) ** 0.5
    return K


# Bad results
def f_corr(X):
    corr_dists = pdist(X, "correlation")

    K = squareform(corr_dists)

    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    return eigvals[-1], K


def f_kernel_corr(X):
    corr_dists = pdist(X, "correlation")

    mat = squareform(corr_dists)
    sigma = max(corr_dists)
    K = np.exp(-2 * mat / sigma)

    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    return eigvals[-1], K


def f_kernel_mahal(X):
    mat = mahal_sqf(X)
    sigma = max(mat)
    K = np.exp(-2 * mat / sigma)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    return eigvals[-1], K
