import numpy as np
from sklearn.metrics import silhouette_score, jaccard_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prf


def silhouette(u, labels, k=None):
    if k:
        s_score = silhouette_score(u[:, :k], labels)
        return s_score
    s_score = silhouette_score(u[:, :], labels)
    return s_score


def f_measure4(true, labels_pred, avg="micro"):

    maxf = prf(true, labels_pred, average=avg)[2]
    # 1243
    labels_pred1 = np.where(labels_pred == 4, 11, labels_pred)
    labels_pred1 = np.where(labels_pred1 == 3, 4, labels_pred1)
    labels_pred1 = np.where(labels_pred1 == 11, 3, labels_pred1)
    maxf = max(maxf, prf(true, labels_pred1, average=avg)[2])
    # 1324
    labels_pred2 = np.where(labels_pred == 3, 11, labels_pred)
    labels_pred2 = np.where(labels_pred2 == 2, 3, labels_pred2)
    labels_pred2 = np.where(labels_pred2 == 11, 2, labels_pred2)
    maxf = max(maxf, prf(true, labels_pred2, average=avg)[2])

    # 1342
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred3 == 22, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 2, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])

    # 1423
    labels_pred4 = np.where(labels_pred == 2, 22, labels_pred)
    labels_pred4 = np.where(labels_pred == 3, 33, labels_pred4)
    labels_pred4 = np.where(labels_pred == 4, 44, labels_pred4)

    labels_pred4 = np.where(labels_pred4 == 22, 4, labels_pred4)
    labels_pred4 = np.where(labels_pred4 == 33, 2, labels_pred4)
    labels_pred4 = np.where(labels_pred4 == 44, 3, labels_pred4)
    maxf = max(maxf, prf(true, labels_pred4, average=avg)[2])

    # 1432
    labels_pred5 = np.where(labels_pred == 2, 22, labels_pred)
    labels_pred5 = np.where(labels_pred == 4, 44, labels_pred5)

    labels_pred5 = np.where(labels_pred5 == 22, 4, labels_pred5)
    labels_pred5 = np.where(labels_pred5 == 44, 2, labels_pred5)
    maxf = max(maxf, prf(true, labels_pred5, average=avg)[2])

    # 2134
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 4, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])

    # 2143
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 3, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])

    # 2314
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 4, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 2341
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 1, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 2413
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 3, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 2431
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 1, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 3124
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 2, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 3142
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 2, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 3214
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 4, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 3241
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 1, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 3412
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 2, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 3421
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 1, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 4123
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 3, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 4132
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 2, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 4213
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 3, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 4231
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 1, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 4312
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 1, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 2, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    # 4321
    labels_pred3 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred3 = np.where(labels_pred == 2, 22, labels_pred3)
    labels_pred3 = np.where(labels_pred == 3, 33, labels_pred3)
    labels_pred3 = np.where(labels_pred == 4, 44, labels_pred3)

    labels_pred3 = np.where(labels_pred == 11, 4, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 22, 3, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 33, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 44, 1, labels_pred3)
    maxf = max(maxf, prf(true, labels_pred3, average=avg)[2])
    return [1, 1, maxf]


def f_measure3(true, labels_pred, avg="micro"):
    # replacing 1 and 2
    labels_pred1 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred1 = np.where(labels_pred1 == 2, 1, labels_pred1)
    labels_pred1 = np.where(labels_pred1 == 11, 2, labels_pred1)
    # replacing 1 and 3
    labels_pred2 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred2 = np.where(labels_pred2 == 3, 1, labels_pred2)
    labels_pred2 = np.where(labels_pred2 == 11, 3, labels_pred2)
    # replacing 2 and 3
    labels_pred3 = np.where(labels_pred == 2, 11, labels_pred)
    labels_pred3 = np.where(labels_pred3 == 3, 2, labels_pred3)
    labels_pred3 = np.where(labels_pred3 == 11, 3, labels_pred3)
    # perm1
    labels_pred4 = np.where(labels_pred == 2, 22, labels_pred)
    labels_pred4 = np.where(labels_pred == 1, 11, labels_pred4)
    labels_pred4 = np.where(labels_pred == 3, 10, labels_pred4)

    labels_pred4 = np.where(labels_pred4 == 22, 1, labels_pred4)
    labels_pred4 = np.where(labels_pred4 == 11, 3, labels_pred4)
    labels_pred4 = np.where(labels_pred4 == 10, 2, labels_pred4)

    # perm2
    labels_pred5 = np.where(labels_pred == 2, 22, labels_pred)
    labels_pred5 = np.where(labels_pred == 1, 11, labels_pred5)
    labels_pred5 = np.where(labels_pred == 3, 10, labels_pred5)

    labels_pred5 = np.where(labels_pred5 == 22, 3, labels_pred5)
    labels_pred5 = np.where(labels_pred5 == 11, 2, labels_pred5)
    labels_pred5 = np.where(labels_pred5 == 10, 1, labels_pred5)
    p_r_f_s = max(
        prf(true, labels_pred, average=avg),
        prf(true, labels_pred1, average=avg),
        prf(true, labels_pred2, average=avg),
        prf(true, labels_pred3, average=avg),
        prf(true, labels_pred4, average=avg),
        prf(true, labels_pred5, average=avg),
    )
    return [0, 0, p_r_f_s[2]]


def f_measure2(true, labels_pred, avg="micro"):
    labels_pred1 = np.where(labels_pred == 1, 11, labels_pred)
    labels_pred1 = np.where(labels_pred1 == 2, 1, labels_pred1)
    labels_pred1 = np.where(labels_pred1 == 11, 2, labels_pred1)
    p_r_f_s = max(
        prf(true, labels_pred, average=avg), prf(true, labels_pred1, average=avg)
    )
    return [0, 0, p_r_f_s[2]]


def f_measure(tr, labels, k, avg="micro"):
    if k == 2:
        arr = list(f_measure2(tr, labels, avg))
    elif k == 3:
        arr = list(f_measure3(tr, labels, avg))
    elif k == 4:
        arr = f_measure4(tr, labels, avg)
    return arr
