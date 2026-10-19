import numpy as np
def library(x):
    N=len(x); s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1)
    M,S,T=s[:,0],s[:,1],s[:,2]; am,as_=o[:,0],o[:,1]
    L={'M':M,'S':S,'T':T,'M+S':M+S,'Mlag':M*(9-am),'Slag':S*(9-as_),'lagM':9.-am,'lagS':9.-as_,
       'MSlag':(M+S)*(9-np.minimum(am,as_)),'sum':x.sum(1),'relusum':np.maximum(x,0).sum(1),
       'Mearly':M*(am<=4),'Searly':S*(as_<=4),'Mlag2':M*(9-am)**2,'Slag2':S*(9-as_)**2}
    for t in range(10): L[f'x{t}']=x[:,t]
    for t in range(10): L[f'o{t}']=x[:,t]*((am!=t)&(as_!=t))
    s8=np.sort(x[:,:9],1)[:,::-1]; s7=np.sort(x[:,:8],1)[:,::-1]
    L['m1_9']=s8[:,0];L['m2_9']=s8[:,1];L['m1_8']=s7[:,0];L['m2_8']=s7[:,1]
    return L
