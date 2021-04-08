import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn import manifold, datasets

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score# jaccard_score, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prf

from sklearn.cluster import SpectralClustering

def read_LGG():
    x1=pd.read_csv("/hdd/Ztudy/BTP/code/CoALa/.data/LGG/mDNA.csv",sep=",",header=None).T.to_numpy()[:,1:]
    x2=pd.read_csv("/hdd/Ztudy/BTP/code/CoALa/.data/LGG/miRNA2.csv",sep=",").T.to_numpy()[:,1:]
    x3=pd.read_csv("/hdd/Ztudy/BTP/code/CoALa/.data/LGG/RNA.csv",sep=",",header=None).T.to_numpy()
    x4=pd.read_csv("/hdd/Ztudy/BTP/code/CoALa/.data/LGG/RPPA267",sep=",").T.to_numpy()
    tr=pd.read_csv("//hdd/Ztudy/BTP/code/CoALa/.data/LGG/labels267.csv",sep=",").to_numpy()
    tr=tr[:,1].astype("int32")
    name="LGG"
    return x1,x2,x3,x4,tr,3,name
    

def read_STAD():
    x1 = pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/STAD/miRNA223", sep=" ").to_numpy()
    x2 = pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/STAD/mRNA223", sep=" ").to_numpy()
    tr = pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/STAD/GT", sep="\t")["class"]
    tr = tr.astype("int32")
    name = "STAD"
    k = 4
    M = 2
    return x1, x2, tr, k, name

    
def read_OV():
    x1=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/OV/mirna_474",sep=",",header=None).to_numpy()
    x2=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/OV/mrna_2k",sep=",",header=None).to_numpy()
    tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/OV/GT",sep=",",header=None)[1].to_numpy()
    tr=tr.astype("int32")
    name="LGG"
    return x1,x2,tr,2,name

def read_CESC():
    x1=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/CESC/mDNA124",sep=" ").to_numpy()
    x2=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/CESC/miRNA124",sep=" ").to_numpy()
    x3=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/CESC/RNA124",sep=" ").to_numpy()
    x4=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/CESC/RPPA124",sep=" ").to_numpy()
    tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/CESC/GT",sep=" ",header=None).to_numpy()
    tr=tr[:,1].astype('int32')
    x2[np.where(x2==0)]=1
    x3[np.where(x3==0)]=1
    x2[:,:]=np.log(x2)
    x3[:,:]=np.log(x3)
    name="CESC"
    return x1,x2,x3,x4,tr,3,name

def read_BRCA():
    x1=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/BRCA/mDNA398",sep=" ").to_numpy()
    x2=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/BRCA/miRNA398",sep=" ").to_numpy()
    x3=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/BRCA/RNA398",sep=" ").to_numpy()
    x4=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/BRCA/RPPA398",sep=" ").to_numpy()
    tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/BRCA/GT",sep=" ",header=None).to_numpy()[:,1]    
    tr=tr.astype('int32')
    x2[np.where(x2==0)]=1
    x3[np.where(x3==0)]=1
    x2[:,:]=np.log(x2)
    x3[:,:]=np.log(x3)
    name="BRCA"
    return x1,x2,x3,x4,tr,4,name

def spectral(X, n_components):
    sq_dists = pdist(X, 'sqeuclidean')

    mat_sq_dists = squareform(sq_dists)
    sigma_sq=max(sq_dists)
    
    K = np.exp(-mat_sq_dists/sigma_sq)
    
    D = np.zeros((len(K[0]), len(K[0])))
    for x in range(0, len(D[0])):
        D[x][x] = np.sum(K[x])
    D_diag = D.diagonal()
    D_diag = 1/D_diag**1/2
    Dd = np.zeros((len(D),len(D)))
    np.fill_diagonal(Dd, D_diag)
    prod = Dd.dot(K)
    L = np.identity(len(D[0]), dtype=float) + prod.dot(Dd) 
#     L=Dd-K
    eigvals, eigvecs = eigh(L)
#     eigvals, eigvecs = eigh(K)

    X_pc = np.column_stack((eigvals[i]*eigvecs[:,i] for i in range(1,n_components+1)))
#     X_pc = manifold.spectral_embedding(adjacency=K, 
    labels=KMeans(n_clusters=n_components, random_state=0).fit(X_pc[:,:]).predict(X_pc[:,:])  
   
    if n_components==2:
        arr=list(f_measure2(tr,labels))
    elif n_components==3:
        arr=list(f_measure(tr,labels))
    s_score = silhouette_score(X_pc[:,:], labels)
#     arr.append(s_score)
    arr[3]=s_score
    return arr,labels

def f_measure3(true, labels_pred,avg='micro'):
    # replacing 1 and 2
    labels_pred1=np.where(labels_pred==1, 11, labels_pred)
    labels_pred1=np.where(labels_pred1==2, 1, labels_pred1)
    labels_pred1=np.where(labels_pred1==11, 2, labels_pred1)
    # replacing 1 and 3
    labels_pred2=np.where(labels_pred==1, 11, labels_pred)
    labels_pred2=np.where(labels_pred2==3, 1, labels_pred2)
    labels_pred2=np.where(labels_pred2==11, 3, labels_pred2)
    # replacing 2 and 3
    labels_pred3=np.where(labels_pred==2, 11, labels_pred)
    labels_pred3=np.where(labels_pred3==3, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==11, 3, labels_pred3)
    #perm1
    labels_pred4=np.where(labels_pred==2, 22, labels_pred)
    labels_pred4=np.where(labels_pred==1, 11, labels_pred4)
    labels_pred4=np.where(labels_pred==3, 10, labels_pred4)

    labels_pred4=np.where(labels_pred4==22, 1, labels_pred4)
    labels_pred4=np.where(labels_pred4==11, 3, labels_pred4)
    labels_pred4=np.where(labels_pred4==10, 2, labels_pred4)

    #perm2
    labels_pred5=np.where(labels_pred==2, 22, labels_pred)
    labels_pred5=np.where(labels_pred==1, 11, labels_pred5)
    labels_pred5=np.where(labels_pred==3, 10, labels_pred5)

    labels_pred5=np.where(labels_pred5==22, 3, labels_pred5)
    labels_pred5=np.where(labels_pred5==11, 2, labels_pred5)
    labels_pred5=np.where(labels_pred5==10, 1, labels_pred5)
    p_r_f_s = max(prf(true, labels_pred, average=avg),
                   prf(true, labels_pred1, average=avg),
                   prf(true, labels_pred2, average=avg),
                   prf(true, labels_pred3, average=avg),
                   prf(true, labels_pred4, average=avg),
                   prf(true, labels_pred5, average=avg))
#     a_score = max(accuracy_score(true, labels_pred),
#                    accuracy_score(true, labels_pred1),
#                    accuracy_score(true, labels_pred2),
#                    accuracy_score(true, labels_pred3),
#                    accuracy_score(true, labels_pred4),
#                    accuracy_score(true, labels_pred5))
#     j_score = max(jaccard_score(true, labels_pred, average='micro'),
#                jaccard_score(true, labels_pred1, average='micro'),
#                jaccard_score(true, labels_pred2, average='micro'),
#                jaccard_score(true, labels_pred3, average='micro'),
#                jaccard_score(true, labels_pred4, average='micro'),
#                jaccard_score(true, labels_pred5, average='micro'))
    return p_r_f_s

def f_measure2(true, labels_pred, avg='micro'):
    labels_pred1=np.where(labels_pred==1, 11, labels_pred)
    labels_pred1=np.where(labels_pred1==2, 1, labels_pred1)
    labels_pred1=np.where(labels_pred1==11, 2, labels_pred1)
    p_r_f_s = max(prf(true, labels_pred, average=avg),
                   prf(true, labels_pred1, average=avg))
#     a_score = max(accuracy_score(true, labels_pred),
#                    accuracy_score(true, labels_pred1))
    
#     j_score = max(jaccard_score(true, labels_pred, average='micro'),
#                jaccard_score(true, labels_pred1, average='micro'))
    return p_r_f_s

def f_measure4(true, labels_pred,avg='micro'):    
    
    maxf=prf(true,labels_pred, average=avg)[2]
# 1243
    labels_pred1=np.where(labels_pred==4, 11, labels_pred)
    labels_pred1=np.where(labels_pred1==3, 4, labels_pred1)
    labels_pred1=np.where(labels_pred1==11, 3, labels_pred1)
    maxf=max(maxf,prf(true,labels_pred1,average=avg)[2])
# 1324
    labels_pred2=np.where(labels_pred==3, 11, labels_pred)
    labels_pred2=np.where(labels_pred2==2, 3, labels_pred2)
    labels_pred2=np.where(labels_pred2==11, 2, labels_pred2)
    maxf=max(maxf,prf(true,labels_pred2,average=avg)[2])

# 1342
    labels_pred3=np.where(labels_pred==2, 22, labels_pred)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred3==22, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 2, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
    
# 1423
    labels_pred4=np.where(labels_pred==2, 22, labels_pred)
    labels_pred4=np.where(labels_pred==3, 33, labels_pred4)
    labels_pred4=np.where(labels_pred==4, 44, labels_pred4)

    labels_pred4=np.where(labels_pred4==22, 4, labels_pred4)
    labels_pred4=np.where(labels_pred4==33, 2, labels_pred4)
    labels_pred4=np.where(labels_pred4==44, 3, labels_pred4)
    maxf=max(maxf,prf(true,labels_pred4,average=avg)[2])
    
# 1432
    labels_pred5=np.where(labels_pred==2, 22, labels_pred)
    labels_pred5=np.where(labels_pred==4, 44, labels_pred5)

    labels_pred5=np.where(labels_pred5==22, 4, labels_pred5)
    labels_pred5=np.where(labels_pred5==44, 2, labels_pred5)
    maxf=max(maxf,prf(true,labels_pred5,average=avg)[2])
    
# 2134
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 4, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
    

# 2143
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 3, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
    
# 2314
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred3==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 4, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 2341
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 1, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 2413
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 3, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 2431
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 1, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 3124
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 2, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 3142
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 2, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 3214
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 4, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 3241
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 1, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 3412
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 2, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 3421
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 1, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 4123
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 3, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 4132
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 2, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 4213
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 3, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 4231
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 1, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 4312
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 1, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 2, labels_pred3)
    maxf=max(maxf,prf(true,labels_pred3,average=avg)[2])
# 4321
    labels_pred3=np.where(labels_pred==1, 11, labels_pred)
    labels_pred3=np.where(labels_pred==2, 22, labels_pred3)
    labels_pred3=np.where(labels_pred==3, 33, labels_pred3)
    labels_pred3=np.where(labels_pred==4, 44, labels_pred3)

    labels_pred3=np.where(labels_pred==11, 4, labels_pred3)
    labels_pred3=np.where(labels_pred3==22, 3, labels_pred3)
    labels_pred3=np.where(labels_pred3==33, 2, labels_pred3)
    labels_pred3=np.where(labels_pred3==44, 1, labels_pred3)
    maxf=max(maxf,prf(true, labels_pred3,average=avg)[2])
#     p_r_f_s = max(prf(true, labels_pred, average=avg),
#                    prf(true, labels_pred1, average=avg),
#                    prf(true, labels_pred2, average=avg),
#                    prf(true, labels_pred3, average=avg),
#                    prf(true, labels_pred4, average=avg),
#                    prf(true, labels_pred5, average=avg))
    return [1,1,maxf]
def f_measure(tr, labels, k,avg='micro'):
    if k==2:
        arr=list(f_measure2(tr,labels,avg))
    elif k==3:
        arr=list(f_measure3(tr,labels,avg))
    elif k==4:
        arr=f_measure4(tr,labels,avg)
    return arr 

x1,x2,tr,k,name=read_OV()
# x1,x2,x3,x4,tr,k,name=read_LGG()
X=x1
sq_dists = pdist(X, 'sqeuclidean')

mat_sq_dists = squareform(sq_dists)
sigma_sq=max(sq_dists)

K = np.exp(-0.5*mat_sq_dists/sigma_sq)

D = np.zeros((len(K[0]), len(K[0])))
for x in range(0, len(D[0])):
    D[x][x] = np.sum(K[x])
D_diag = D.diagonal()
D_diag = 1/D_diag**1/2
Dd = np.zeros((len(D),len(D)))
np.fill_diagonal(Dd, D_diag)

L = np.identity(len(D[0]), dtype=float) - Dd.dot(K).dot(Dd) 
# L=Dd-K
eigvals, u = eigh(L)
#     eigvals, eigvecs = eigh(K)

# X_pc = np.column_stack((eigvals[i]*eigvecs[:,i] for i in range(1,n_components+1)))
#     X_pc = manifold.spectral_embedding(adjacency=K, 
labels=KMeans(n_clusters=k, random_state=0).fit(u[:,:k]).predict(u[:,:k])+1  

arr=f_measure(tr,labels,k)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(14,-65)
s=20
# ax.scatter3D(u[np.where(tr==1),0],u[np.where(tr==1),1],s=s,marker='o')
# ax.scatter3D(u[np.where(tr==2),0],u[np.where(tr==2),1],s=s,marker='v')


p1={}
p1['red']='lightcoral'
p1['blue']='cornflowerblue'
p1['green']='yellowgreen'
p1['purple']='orchid'

p2={}
p2['red']='r'
p2['blue']='b'
p2['green']='g'
p2['purple']='black'


# ax.scatter3D(u[np.where(tr==1),0],u[np.where(tr==1),1],u[np.where(tr==1),2],c=p2['red'],s=s)
# ax.scatter3D(u[np.where(tr==2),0],u[np.where(tr==2),1],u[np.where(tr==2),2],c=p2['green'],s=s)
# ax.scatter3D(u[np.where(tr==3),0],u[np.where(tr==3),1],u[np.where(tr==3),2],c=p2['blue'],s=s)
# ax.scatter3D(u[np.where(tr==4),0],u[np.where(tr==4),1],u[np.where(tr==4),2],c=p2['purple'],s=s)

# ax.set_xlim([round(min(u[:,0]),2), round(max(u[:,0]),2)])
# ax.set_ylim([round(min(u[:,1]),2), round(max(u[:,1]),2)])
# ax.set_zlim([round(min(u[:,2]),2), round(max(u[:,2]),2)])
# ax.scatter3D(0,0,0,s=0)

plt.scatter(u[np.where(tr==1),0],u[np.where(tr==1),1],s=s)
plt.scatter(u[np.where(tr==2),0],u[np.where(tr==2),1],s=s)
plt.scatter(u[np.where(tr==3),0],u[np.where(tr==3),1],s=s)
plt.scatter(u[np.where(tr==4),0],u[np.where(tr==4),1],s=s)


# plt.axis('off')
from matplotlib import rcParams
rcParams['figure.dpi'] = 1000
print("max:",[max(u[:,0]),max(u[:,1]),max(u[:,2])])
print("min:",[min(u[:,0]),min(u[:,1]),min(u[:,2])])
plt.show()