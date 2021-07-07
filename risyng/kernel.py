import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


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
