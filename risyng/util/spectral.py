from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np
from sklearn.cluster import KMeans
from util.metrics import silhouette
from util.metrics import f_measure


def spectral(X, n_components):
    pass


#     sq_dists = pdist(X, 'sqeuclidean')

#     mat_sq_dists = squareform(sq_dists)
#     sigma_sq=max(sq_dists)

#     K = np.exp(-2*mat_sq_dists/sigma_sq)

#     D = np.zeros((len(K[0]), len(K[0])))
#     for x in range(0, len(D[0])):
#         D[x][x] = np.sum(K[x])
#     D_diag = D.diagonal()
#     D_diag = 1/D_diag**1/2
#     Dd = np.zeros((len(D),len(D)))
#     np.fill_diagonal(Dd, D_diag)
# #     prod = Dd.dot(K)
# #     L = np.identity(len(D[0]), dtype=float) - prod.dot(Dd)
#     L=Dd-K
#     eigvals, eigvecs = eigh(L)
# #     eigvals, eigvecs = eigh(K)

#     X_pc = np.column_stack((eigvals[i]*eigvecs[:,i] for i in range(1,n_components+1)))
# #     X_pc = manifold.spectral_embedding(adjacency=K,
#     labels=KMeans(n_clusters=n_components, random_state=0).fit(X_pc[:,:]).predict(X_pc[:,:])+1

#     if n_components==2:
#         arr=list(f_measure2(tr,labels))
#     elif n_components==3:
#         arr=list(f_measure(tr,labels))
#     s_score = silhouette(X_pc[:,:], labels)
# #     arr.append(s_score)
#     arr[3]=s_score
#     return arr,labels
