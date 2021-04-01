import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# BRCA
u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-BRCA",sep=' ',header=None).to_numpy()
tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/BRCA/GT",sep=" ",header=None).to_numpy()[:,1]    
tr=tr.astype('int32')

plt.scatter(u[np.where(tr==1),0],u[np.where(tr==1),1],s=6.5)
plt.scatter(u[np.where(tr==2),0],u[np.where(tr==2),1],s=6.5)
plt.scatter(u[np.where(tr==3),0],u[np.where(tr==3),1],s=6.5)
plt.scatter(u[np.where(tr==4),0],u[np.where(tr==4),1],s=6.5)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/BRCA.png",dpi=100)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/BRCA.eps",dpi=100)

# LGG
data="LGG"
u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
tr=pd.read_csv("//hdd/Ztudy/BTP/code/CoALa/.data/LGG/labels267.csv",sep=",").to_numpy()
tr=tr[:,1].astype("int32")

plt.scatter(u[np.where(tr==1),0],u[np.where(tr==1),1],s=6.5)
plt.scatter(u[np.where(tr==2),0],u[np.where(tr==2),1],s=6.5)
plt.scatter(u[np.where(tr==3),0],u[np.where(tr==3),1],s=6.5)
# plt.scatter(u[np.where(tr==4),0],u[np.where(tr==4),1],s=6.5)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/"+data+".png",dpi=100)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/"+data+".eps",dpi=100)

# CESC
data="CESC"
u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/CESC/GT",sep=" ",header=None).to_numpy()
tr=tr[:,1].astype('int32')

plt.scatter(u[np.where(tr==1),0],u[np.where(tr==1),1],s=6.5)
plt.scatter(u[np.where(tr==2),0],u[np.where(tr==2),1],s=6.5)
plt.scatter(u[np.where(tr==3),0],u[np.where(tr==3),1],s=6.5)
# plt.scatter(u[np.where(tr==4),0],u[np.where(tr==4),1],s=6.5)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/"+data+".png",dpi=100)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/"+data+".eps",dpi=100)


#STAD
data="STAD"
u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
tr = pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/STAD/GT", sep="\t")["class"]
tr = tr.astype("int32")

plt.scatter(u[np.where(tr==1),0],u[np.where(tr==1),1],s=6.5)
plt.scatter(u[np.where(tr==2),0],u[np.where(tr==2),1],s=6.5)
# plt.scatter(u[np.where(tr==3),0],u[np.where(tr==3),1],s=6.5)
# plt.scatter(u[np.where(tr==4),0],u[np.where(tr==4),1],s=6.5)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/"+data+".png",dpi=100)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/"+data+".eps",dpi=100)

#OVdata="OV"
u = pd.read_csv("/hdd/Ztudy/BTP/code/algoTrials/intWA/dat-"+data,sep=' ',header=None).to_numpy()
tr=pd.read_csv("/hdd/projects/coala-research/CoALa/Data Sets/OV/GT",sep=",",header=None)[1].to_numpy()
tr=tr.astype("int32")

plt.scatter(u[np.where(tr==1),0],u[np.where(tr==1),1],s=6.5)
plt.scatter(u[np.where(tr==2),0],u[np.where(tr==2),1],s=6.5)
# plt.scatter(u[np.where(tr==3),0],u[np.where(tr==3),1],s=6.5)
# plt.scatter(u[np.where(tr==4),0],u[np.where(tr==4),1],s=6.5)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/"+data+".png",dpi=100)
plt.savefig("/hdd/Ztudy/BTP/code/algoTrials/plotting/plots/"+data+".eps",dpi=100)