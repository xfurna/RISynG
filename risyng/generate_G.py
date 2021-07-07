from sklearn.cluster import KMeans
from util import read
import risyng_algo
import integrate
from util.metrics import silhouette as silhouette_score
from util.metrics import f_measure
import numpy as np
import argparse


def caller(X, b, k, tr):
    s, u, G, kern = risyng_algo.RISynG(X, b)
    labels = (
        KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=3242)
        .fit(u[:, :k])
        .predict(u[:, :k])
        + 1
    )
    s_score = silhouette_score(u[:, :k], labels)
    arr = f_measure(tr, labels, k)
    return s_score, arr, labels, G, kern


def execute(data="LGG", file=1, b=-1):
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
    if b != "N":
        s_score, arr, labels, G, kern = caller(X, b, k, tr)
        sil.append(s_score)
        f.append(arr[2])
        sil = np.round(sil, 3)
        f = np.round(f, 3)

        return sil[0], f[0], labels, kern, G, b

    for i in range(11):
        s_score, arr, labels, G, kern = caller(X, i / 10, k, tr)
        sil.append(s_score)
        f.append(arr[2])
    inds = [(f[i], sil[j], j / 10) for j in range(11)]
    inds.sort()
    # print("inds:  ",inds)
    sil = np.round(sil, 3)
    f = np.round(f, 3)
    t = 0
    s_score, arr, labels, G, kern = caller(X, inds[t][2], k, tr)
    return s_score, arr[2], labels, kern, G, inds[t][2]


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--fuse", "-f", nargs="?", default="n", help="Pass 'y' for fusion."
    )
    args.add_argument("--dataset", "-d", nargs="?", help="Which dataset?")
    args.add_argument(
        "--modality", "-m", nargs="?", help="Which modality? (an integer)"
    )
    args.add_argument(
        "--bval",
        "-b",
        nargs="?",
        default="N",
        help="At what b? (a number, or 'N' for default)",
    )
    args.add_argument("--order", "-o", nargs="?", help="In what oder?")
    args.add_argument("--cluster", "-k", nargs="?", help="Number of clusters")

    pargs = args.parse_args()

    if pargs.fuse == "y":
        integrate.main(pargs)
        return
    if pargs.bval != "N":
        b = float(pargs.bval)
    else:
        b = "N"
    sil, f, labels, kern, G, param = execute(pargs.dataset, int(pargs.modality), b=b)
    filename = "../Gmat/mod-" + str(pargs.modality) + "-" + pargs.dataset
    np.savetxt(filename, G)
    print("[DATA SAVED] ", filename, "\n")
    print("b,\tsil,\tf")
    print(param, ",", sil, ",", f)
    pass


if __name__ == "__main__":
    main()
