import kernel
import laplacian as lp
import numpy as np


def RISynG(X, b):
    y, kern = kernel.f_kernel_corr(X)
    e, lap = lp.f_lap(X)
    I = np.identity(kern.shape[0])
    N = kern.shape[0]

    G = (1 - b) * (I - kern) + b * lap

    s, u = np.linalg.eigh(G)
    return s, u, G, kernel
