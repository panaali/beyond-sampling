import numpy as np
def feats(x):
  F={}
  for k in range(10): F[f'x{k}']=x[:,k]
  M=np.maximum.accumulate(x,1); S=np.maximum.accumulate(x[:,::-1],1)[:,::-1]
  for k in range(1,10): F[f'M{k}']=M[:,k]   # max x0..xk
  for k in range(1,9): F[f'S{k}']=S[:,k]    # max xk..x9
  srt=-np.sort(-x,1)
  for k in range(4): F[f'srt{k}']=srt[:,k]
  for k in range(10):
    F[f'mx_no{k}']=np.max(np.delete(x,k,1),1)
  F['sum']=x.sum(1)
  return F
