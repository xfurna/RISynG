import pandas as pd
import argparse
import numpy as np
from scipy.linalg import eigh


def orthog(U, eps=1e-15):
    n = len(U[0])
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]     
        coeff_vec = np.dot(prev_basis, V[i].T)  
      
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if np.linalg.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.
        else:
            V[i] /= np.linalg.norm(V[i])
    return V.T

subpath="Gmat/mod-"
classes = {'BRCA':4,'CESC':3,'LGG':3,'OV':2,'STAD':2}

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', '-d', help="Which dataset?")
    args.add_argument('--order', '-o', help="In what oder?")
    pargs = args.parse_args() 
    data = pargs.dataset

    WA=np.zeros((len(data[0]),k)) #for wt sum of ortho u

    for i in pargs.order:
        file = subpath+ i + "-" + data
        X = pd.read_csv(file,delimiter=" ",header=None).to_numpy()
        s,u=eigh(X[int(i)])
        k = classes[data]
        
        U=u[:,:k]
        I=np.dot(A.T,U)
        P=np.dot(A, I)
        Q=U-P
        G=orthog(Q)
        wg=G*G