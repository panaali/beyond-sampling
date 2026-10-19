import numpy as np
from feats2 import feats as f2
def feats(x):
    F=f2(x); W=x[:,:9]
    o=np.argsort(-x,1); am,a2=o[:,0],o[:,1]; o9=np.argsort(-W,1); b1,b2=o9[:,0],o9[:,1]
    s=-np.sort(-W,1); g12=s[:,0]-s[:,1]; g23=s[:,1]-s[:,2]
    F['g12']=g12; F['g23']=g23
    for j in range(10):
        F['amA%d'%j]=(am==j)*1.; F['a2A%d'%j]=(a2==j)*1.
    for j in range(9):
        F['am%d*g12'%j]=(b1==j)*g12; F['a2%d*g23'%j]=(b2==j)*g23
    F['pos_am9']=b1*1.;F['pos_a29']=b2*1.;F['pos_amA']=am*1.;F['pos_a2A']=a2*1.
    F['a2A_even']=(a2%2==0)*1.; F['a29_even']=(b2%2==0)*1.; F['am9_even']=(b1%2==0)*1.; F['amA_even']=(am%2==0)*1.
    return F
