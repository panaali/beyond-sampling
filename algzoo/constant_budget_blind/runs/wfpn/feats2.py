import numpy as np
from feats import feats
def feats2(x):
  F=feats(x)
  srt=-np.sort(-x,1); M=np.maximum.accumulate(x,1)
  s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
  for k in range(10):
    F[f'G{k}']=M[:,k]-x[:,k]
    F[f'top-x{k}']=srt[:,0]-x[:,k]
    F[f'sec-x{k}']=srt[:,1]-x[:,k]
    F[f'relu(sec-x{k})']=np.maximum(srt[:,1]-x[:,k],0)
    F[f'relu(x{k})']=np.maximum(x[:,k],0)
    F[f'relu(-x{k})']=np.maximum(-x[:,k],0)
    F[f'top*[am={k}]']=srt[:,0]*(am==k)
    F[f'sec*[a2={k}]']=srt[:,1]*(a2==k)
    F[f'top*[am>={k}]']=srt[:,0]*(am>=k)
    F[f'top*[am<{k}]']=srt[:,0]*(am<k)
  for k in range(1,10):
    F[f'relu(M{k-1}-x{k})']=np.maximum(M[:,k-1]-x[:,k],0)
    F[f'relu(x{k}-M{k-1})']=np.maximum(x[:,k]-M[:,k-1],0)
  F['range']=srt[:,0]-srt[:,-1]
  F['absmean']=np.abs(x).mean(1)
  return F
