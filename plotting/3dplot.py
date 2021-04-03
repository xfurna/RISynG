import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D

data="LGG"
tr=pd.read_csv("//hdd/Ztudy/BTP/code/CoALa/.data/LGG/labels267.csv",sep=",").to_numpy()
tr=tr[:,1].astype("int32")
u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
# SNF: u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/LGG/SNF_vector_lgg",sep=' ',header=None).to_numpy()
# coala: u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/LGG/JointU_LGG.txt",sep=' ',header=None).to_numpy()
# u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/LGG/sure_LGG.txt",sep=' ',header=None).to_numpy()


# data="CESC"
# tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/CESC/GT",sep=" ",header=None).to_numpy()
# tr=tr[:,1].astype('int32')
# risyng: u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
# SNF: u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/CESC/SNF_vector_cesc",sep=' ',header=None).to_numpy()
# coala: u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/CESC/coala_CESC.txt",sep=' ',header=None).to_numpy()
# sure: u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/CESC/sure_CESC.txt",sep=' ',header=None).to_numpy()


# data="BRCA"
# tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/BRCA/GT",sep=" ",header=None).to_numpy()[:,1]    
# tr=tr.astype('int32')
# risyng: u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
# snf: u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/BRCA/SNF_vector_brca",sep=' ',header=None).to_numpy()
# caoal:u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/BRCA/coala_BRCA.txt",sep=' ',header=None).to_numpy()
# sure u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/BRCA/sure_BRCA.txt",sep=' ',header=None).to_numpy()


# data="STAD"
# tr = pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/STAD/GT", sep="\t")["class"]
# tr = tr.astype("int32")
# risyng: u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
# snf: u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/STAD/SNF_vector_stad",sep=' ',header=None).to_numpy()
# u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/STAD/coala_STAD.txt",sep='\t',header=None).to_numpy()
# u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/BRCA/sure_BRCA.txt",sep=' ',header=None).to_numpy()

# data='OV'
# tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/OV/GT",sep=",",header=None)[1].to_numpy()
# tr=tr.astype("int32")
# risyng: u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
# u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/OV/SNF_vector_ov",sep=' ',header=None).to_numpy()
# u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/OV/coala_OV.txt",sep=' ',header=None).to_numpy()
# u=pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/plotting/outfolders/eigen_vectors/OV/sure_OV.txt",sep=' ',header=None).to_numpy()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
s=20
# ax.scatter3D(u[np.where(tr==1),0],u[np.where(tr==1),1],s=s,marker='o')
# ax.scatter3D(u[np.where(tr==2),0],u[np.where(tr==2),1],s=s,marker='v')

c1=[np.sum(u[np.where(tr==1),0])/u[np.where(tr==1),0].shape[1],np.sum(u[np.where(tr==1),1])/u[np.where(tr==1),0].shape[1],np.sum(u[np.where(tr==1),2])/u[np.where(tr==1),0].shape[1]]
c2=[np.sum(u[np.where(tr==2),0])/u[np.where(tr==2),0].shape[1],np.sum(u[np.where(tr==2),1])/u[np.where(tr==2),0].shape[1],np.sum(u[np.where(tr==2),2])/u[np.where(tr==2),0].shape[1]]
c3=[np.sum(u[np.where(tr==3),0])/u[np.where(tr==3),0].shape[1],np.sum(u[np.where(tr==3),1])/u[np.where(tr==3),0].shape[1],np.sum(u[np.where(tr==3),2])/u[np.where(tr==3),0].shape[1]]
if 4 in tr:
    c4=[np.sum(u[np.where(tr==4),0])/u[np.where(tr==4),0].shape[1],np.sum(u[np.where(tr==4),1])/u[np.where(tr==4),0].shape[1],np.sum(u[np.where(tr==4),2])/u[np.where(tr==4),0].shape[1]]

p1={}
p1['red']='lightcoral'
p1['blue']='cornflowerblue'
p1['green']='yellowgreen'
p1['purple']='orchid'

p2={}
p2['red']='r'
p2['blue']='b'
p2['green']='g'
p2['purple']='orchid'


ax.scatter3D(u[np.where(tr==1),0],u[np.where(tr==1),1],u[np.where(tr==1),2],c=p2['red'],s=s,marker='o')
ax.scatter3D(u[np.where(tr==2),0],u[np.where(tr==2),1],u[np.where(tr==2),2],c=p2['green'],s=s,marker='v')
ax.scatter3D(0.1+u[np.where(tr==3),0],u[np.where(tr==3),1],u[np.where(tr==3),2],c=p2['blue'],s=s,marker='s')
ax.scatter3D(u[np.where(tr==4),0],u[np.where(tr==4),1],u[np.where(tr==4),2],c=p2['purple'],s=s,marker='p')

# for (x,y,z) in zip(u[np.where(tr==1),0][0],u[np.where(tr==1),1][0],u[np.where(tr==1),2][0]):
#     plt.plot([x,c1[0]],[y,c1[1]],[z,c1[2]],p1['red'])
# for (x,y,z) in zip(u[np.where(tr==2),0][0],u[np.where(tr==2),1][0],u[np.where(tr==2),2][0]):
#     plt.plot([x,c2[0]],[y,c2[1]],[z,c2[2]],p1['green'])
# for (x,y,z) in zip(u[np.where(tr==3),0][0],u[np.where(tr==3),1][0],u[np.where(tr==3),2][0]):
#     plt.plot([x,c3[0]],[y,c3[1]],[z,c3[2]],p1['blue'])


# plt.scatter(u[np.where(tr==1),0],u[np.where(tr==1),1],s=s,marker='o')
# plt.scatter(u[np.where(tr==2),0],u[np.where(tr==2),1],s=s,marker='v')
# plt.scatter(u[np.where(tr==3),0],u[np.where(tr==3),1],s=s,marker='s')
# plt.scatter(u[np.where(tr==4),0],u[np.where(tr==4),1],s=s,marker='p')
plt.axis('off')
from matplotlib import rcParams
rcParams['figure.dpi'] = 500
plt.show()