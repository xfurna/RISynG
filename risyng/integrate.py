import pandas as pd
import argparse
import numpy as np
from util.read import read_tr as rtr
from scipy.linalg import eigh
from util.metrics import f_measure
from sklearn.cluster import KMeans
from util.metrics import silhouette as silhouette_score


def orthog(U, eps=1e-15):
    n = len(U[0])
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]
        coeff_vec = np.dot(prev_basis, V[i].T)

        V[i] -= np.dot(coeff_vec, prev_basis).T
        if np.linalg.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.0
        else:
            V[i] /= np.linalg.norm(V[i])
    return V.T


subpath = "../Gmat/mod-"
classes = {"BRCA": 4, "CESC": 3, "LGG": 3, "OV": 2, "STAD": 4}


def main(pargs):
    data = pargs.dataset
    k = classes[data]
    X = []

    for chari in list(pargs.order):
        file = subpath + chari + "-" + data
        X.append(pd.read_csv(file, delimiter=" ", header=None).to_numpy())

    WA = np.zeros((len(X[0]), k))

    for j in range(len(list(pargs.order))):
        s, u = eigh(X[j])
        U = u[:, :k]
        I = np.dot(WA.T, U)
        P = np.dot(WA, I)
        Q = U - P
        G = orthog(Q)
        wg = G ** (j + 1)
        WA = WA + wg

    tr = rtr(data)
    labels = (
        KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=3242)
        .fit(WA[:, :])
        .predict(WA[:, :])
        + 1
    )
    int_labels = labels.astype(int)
    s_score = silhouette_score(WA[:, :], labels)
    arr = f_measure(tr, labels, k)
    filename = "../intWA/dat-" + pargs.dataset
    np.savetxt(filename, WA)
    print("[DATA SAVED] ", filename, "\n")

    labelFile = "../labels/labels-" + pargs.dataset
    np.savetxt(labelFile, int_labels)
    print("[LABEL FILE SAVED] ", labelFile, "\n")
    print(
        pargs.dataset + ":  FINAL-SILHOUETTE\t\tFINAL-FSCORE\n", s_score, ",\t", arr[2]
    )
