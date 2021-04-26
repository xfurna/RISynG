import pandas as pd
import numpy as np

datadir = "Datasets/"


def read_LGG():
    x1 = pd.read_csv(datadir + "LGG/mDNA.csv", sep=",", header=None).T.to_numpy()[:, 1:]
    x2 = pd.read_csv(datadir + "LGG/miRNA2.csv", sep=",").T.to_numpy()[:, 1:]
    x3 = pd.read_csv(datadir + "LGG/RNA.csv", sep=",", header=None).T.to_numpy()
    x4 = pd.read_csv(datadir + "LGG/RPPA267", sep=",").T.to_numpy()
    tr = pd.read_csv(datadir + "LGG/labels267.csv", sep=",").to_numpy()
    tr = tr[:, 1].astype("int32")
    name = "LGG"
    k = 3
    return x1, x2, x3, x4, tr, k, name


def read_STAD():
    x1 = pd.read_csv(datadir + "STAD/miRNA223", sep=" ").to_numpy()
    x2 = pd.read_csv(datadir + "STAD/mRNA223", sep=" ").to_numpy()
    tr = pd.read_csv(datadir + "STAD/GT", sep="\t")["class"]
    tr = tr.astype("int32")
    name = "STAD"
    k = 4
    M = 2
    return x1, x2, tr, k, name


def read_OV():
    x1 = pd.read_csv(datadir + "OV/mirna_474", sep=",", header=None).to_numpy()
    x2 = pd.read_csv(datadir + "OV/mrna_2k", sep=",", header=None).to_numpy()
    tr = pd.read_csv(datadir + "OV/GT", sep=",", header=None)[1].to_numpy()
    tr = tr.astype("int32")
    name = "OV"
    k = 2
    return x1, x2, tr, k, name


def read_CESC():
    x1 = pd.read_csv(datadir + "CESC/mDNA124", sep=" ").to_numpy()
    x2 = pd.read_csv(datadir + "CESC/miRNA124", sep=" ").to_numpy()
    x3 = pd.read_csv(datadir + "CESC/RNA124", sep=" ").to_numpy()
    x4 = pd.read_csv(datadir + "CESC/RPPA124", sep=" ").to_numpy()
    tr = pd.read_csv(datadir + "CESC/GT", sep=" ", header=None).to_numpy()
    tr = tr[:, 1].astype("int32")
    x2[np.where(x2 == 0)] = 1
    x3[np.where(x3 == 0)] = 1
    x2[:, :] = np.log(x2)
    x3[:, :] = np.log(x3)
    name = "CESC"
    k = 3
    return x1, x2, x3, x4, tr, k, name


def read_BRCA():
    x1 = pd.read_csv(datadir + "BRCA/mDNA398", sep=" ").to_numpy()
    x2 = pd.read_csv(datadir + "BRCA/miRNA398", sep=" ").to_numpy()
    x3 = pd.read_csv(datadir + "BRCA/RNA398", sep=" ").to_numpy()
    x4 = pd.read_csv(datadir + "BRCA/RPPA398", sep=" ").to_numpy()
    tr = pd.read_csv(datadir + "BRCA/GT", sep=" ", header=None).to_numpy()[:, 1]
    tr = tr.astype("int32")
    x2[np.where(x2 == 0)] = 1
    x3[np.where(x3 == 0)] = 1
    x2[:, :] = np.log(x2)
    x3[:, :] = np.log(x3)
    name = "BRCA"
    k = 4
    return x1, x2, x3, x4, tr, k, name


def read_tr(data):
    if data == "STAD":
        tr = pd.read_csv(datadir + "STAD/GT", sep="\t")["class"]
        tr = tr.astype("int32")
    if data == "LGG":
        tr = pd.read_csv(datadir + "LGG/labels267.csv", sep=",").to_numpy()
        tr = tr[:, 1].astype("int32")
    if data == "BRCA":
        tr = pd.read_csv(datadir + "BRCA/GT", sep=" ", header=None).to_numpy()[:, 1]
        tr = tr.astype("int32")
    if data == "OV":
        tr = pd.read_csv(datadir + "OV/GT", sep=",", header=None)[1].to_numpy()
        tr = tr.astype("int32")
    if data == "CESC":
        tr = pd.read_csv(datadir + "CESC/GT", sep=" ", header=None).to_numpy()
        tr = tr[:, 1].astype("int32")
    return tr
