import numpy as np
def pool(x):
    N=len(x); M=np.maximum.accumulate(x,1); s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
    srt=-np.sort(-x,1); F={}; m=M[:,9]
    odd=(am%2==1)*1.
    F['M9odd']=m*odd; F['M9even']=m*(1-odd)
    for j in range(10):
        F[f'x{j}']=x[:,j]; F[f'M{j}']=M[:,j]; F[f's{j}']=srt[:,j]
        F[f'D{j}']=M[:,j]-x[:,j]; F[f'T{j}']=m-x[:,j]
        F[f'Mam{j}']=m*(am==j); F[f'Mamge{j}']=m*(am>=j)
        F[f'Ma2{j}']=m*(a2==j); F[f'S1a2{j}']=srt[:,1]*(a2==j)
        if j>0:
            F[f'G{j}']=np.maximum(x[:,j]-M[:,j-1],0)
    return F
def pool2(x):
    F=pool(x); m=F['M9']
    F['sumx']=x.sum(1); F['sumE']=x[:,0::2].sum(1); F['sumO']=x[:,1::2].sum(1)
    F['negsum']=np.maximum(-x,0).sum(1); F['possum']=np.maximum(x,0).sum(1)
    for k in range(10):
        F[f'cum{k}']=x[:,k:].sum(1)  # sum of last positions
    return F
def pool3(x):
    F=pool2(x)
    for nm in ['T','D','x']:
        F[nm+'sE']=sum(F[f'{nm}{j}'] for j in range(0,10,2)); F[nm+'sO']=sum(F[f'{nm}{j}'] for j in range(1,10,2))
    m=F['M9']; s=np.argsort(-x,1); am=s[:,0]
    F['M9age']=m*(9-am)
    return F
