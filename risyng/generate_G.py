# from scipy.spatial.distance import pdist, squareform
# from scipy.linalg import eigh
# from sklearn import manifold, datasets

from sklearn.cluster import KMeans

# from sklearn.cluster import SpectralClustering

from util import read
import risyng_algo
import integrate
from util.metrics import silhouette as silhouette_score
from util.metrics import f_measure
import numpy as np


def execute(v=1, data="LGG", file=1, b=-1):
    if data == "LGG":
        x1, x2, x3, x4, tr, k, dat = read.read_LGG()
        nx1 = x1 - x1.mean(0)
        nx2 = x2 - x2.mean(0)
        nx3 = x3 - x3.mean(0)
        nx4 = x4 - x4.mean(0)
        if file == 1:
            X = nx1
        elif file == 2:
            X = nx2
        elif file == 3:
            X = nx3
        elif file == 4:
            X = nx4
    if data == "BRCA":
        x1, x2, x3, x4, tr, k, dat = read.read_BRCA()
        nx1 = x1 - x1.mean(0)
        nx2 = x2 - x2.mean(0)
        nx3 = x3 - x3.mean(0)
        nx4 = x4 - x4.mean(0)
        if file == 1:
            X = nx1
        elif file == 2:
            X = nx2
        elif file == 3:
            X = nx3
        elif file == 4:
            X = nx4
    elif data == "CESC":
        x1, x2, x3, x4, tr, k, dat = read.read_CESC()
        nx1 = x1 - x1.mean(0)
        nx2 = x2 - x2.mean(0)
        nx3 = x3 - x3.mean(0)
        nx4 = x4 - x4.mean(0)
        if file == 1:
            X = nx1
        elif file == 2:
            X = nx2
        elif file == 3:
            X = nx3
        elif file == 4:
            X = nx4
    elif data == "STAD":
        x1, x2, tr, k, dat = read.read_STAD()
        nx1 = x1 - x1.mean(0)
        nx2 = x2 - x2.mean(0)
        if file == 1:
            X = nx1
        else:
            X = nx2
    elif data == "OV":
        x1, x2, tr, k, dat = read.read_OV()
        nx1 = x1 - x1.mean(0)
        nx2 = x2 - x2.mean(0)
        if file == 1:
            X = nx1
        else:
            X = nx2

    sil = []
    f = []
    if b != -1:
        if v == 1:
            s, u, G = risyng_algo.RISynG_v1(X, b)
            kern = X
        elif v == 2:
            s, u, G, kern = risyng_algo.RISynG_v2(X, b)
        elif v == 3:
            s, u, G, kern = risyng_algo.RISynG_v3(X, b)
        elif v == 4:
            s, u, G, kern = risyng_algo.RISynG_v4(X, b)
        labels = (
            KMeans(n_clusters=k, random_state=0).fit(u[:, :k]).predict(u[:, :k]) + 1
        )
        s_score = silhouette_score(u[:, :k], labels)
        arr = f_measure(tr, labels, k)
        sil.append(s_score)
        f.append(arr[2])
        sil = np.round(sil, 3)
        f = np.round(f, 3)
        return sil, f, labels, kern, G

    for i in range(11):
        if v == 1:
            s, u, G = risyng_algo.RISynG_v1(X, i / 10)
            kern = X
        elif v == 2:
            s, u, G, kern = risyng_algo.RISynG_v2(X, i / 10)
        elif v == 3:
            s, u, G, kern = risyng_algo.RISynG_v3(X, i / 10)
        elif v == 4:
            s, u, G, kern = risyng_algo.RISynG_v4(X, i / 10)

        labels = (
            KMeans(n_clusters=k, random_state=0).fit(u[:, :k]).predict(u[:, :k]) + 1
        )
        s_score = silhouette_score(u[:, :k], labels)
        arr = f_measure(tr, labels, k)
        sil.append(s_score)
        f.append(arr[2])
    # print(arr,",",s_score)
    # labels
    # plt.scatter(list(range(11)),sil)
    # plt.scatter(list(range(11)),np.array(f)/1.5)
    sil = np.round(sil, 3)
    f = np.round(f, 3)
    return sil, f, labels, kern, G


import argparse

def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--fuse", "-f", nargs="?", default='n', help="Pass 'y' for fusion."
    )
    args.add_argument("--dataset", "-d", nargs="?", help="Which dataset?")
    args.add_argument("--modality", "-m", nargs="?", help="Which modality? (an integer)")
    args.add_argument(
        "--bval", "-b", nargs="?", default=-1, help="At what b? (a number)"
    )
    args.add_argument("--order", "-o", nargs="?", help="In what oder?")

    pargs = args.parse_args()

    if pargs.fuse=='y':
        integrate.main(pargs)
        return

    b = float(pargs.bval)
    sil, f, labels, kern, G = execute(2, pargs.dataset, int(pargs.modality), b=b)
    for i in range(len(sil)):
        param = i / 10
        if b != -1:
            param = b
            filename = "../Gmat/mod-" + str(pargs.modality) + "-" + pargs.dataset
            np.savetxt(filename, G)
            print("[DATA SAVED] ", filename, "\n")
        print("b,\tsil,\tf")
        print(param, ",", sil[i], ",", f[i])
    pass

if __name__ == "__main__":
    main()