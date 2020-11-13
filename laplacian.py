from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np


def f_lap(X):
    sq_dists = pdist(X, "sqeuclidean")

    mat_sq_dists = squareform(sq_dists)

    sigma = max(sq_dists)
    K = np.exp(-2 * mat_sq_dists / sigma)
    D = np.zeros((len(K[0]), len(K[0])))
    for x in range(0, len(D[0])):
        D[x][x] = np.sum(K[x])
    D_diag = D.diagonal()
    D_diag = D_diag ** (-0.5)
    Dd = np.zeros((len(D), len(D)))
    np.fill_diagonal(Dd, D_diag)
    prod = Dd.dot(K)
    L = np.identity(len(D[0]), dtype=float) - prod.dot(Dd)
    eigvals, eigvecs = eigh(L)
    return eigvals[-1], L
