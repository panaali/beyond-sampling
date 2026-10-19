import numpy as np
def pool(x):
  N=len(x); F={}
  M=np.maximum.accumulate(x,1)
  G=np.zeros((N,10)); U=np.zeros((N,10)); U[:,0]=x[:,0]
  for t in range(1,10):
    G[:,t]=np.maximum(M[:,t-1]-x[:,t],0); U[:,t]=np.maximum(x[:,t]-M[:,t-1],0)
  s=-np.sort(-x,1)
  for k in range(10):
    F[f'x{k}']=x[:,k]; F[f'M{k}']=M[:,k]
  for a in range(10):
    for b in range(a,10):
      if a>=1: F[f'G[{a}:{b}]']=G[:,a:b+1].sum(1)
      if b>a: F[f'x[{a}:{b}]']=x[:,a:b+1].sum(1)
  for k in range(1,5): F[f'top{k}sum']=s[:,:k].sum(1)
  for k in range(2,5): F[f'top{k}sum_0..8']=(-np.sort(-x[:,:9],1))[:,:k].sum(1)
  F['sec']=s[:,1]
  return F
def pool2(x):
  F=pool(x); N=len(x)
  for t in range(1,10):
    p=x[:,:t]; s=-np.sort(-p,1); L=s[:,:min(2,t)].mean(1)
    F[f'g{t}']=np.maximum(L-x[:,t],0); F[f'u{t}']=np.maximum(x[:,t]-L,0); F[f'T{t-1}']=L
  return F
def pool3(x):
  F=pool2(x)
  g=np.stack([np.zeros(len(x))]+[F[f'g{t}'] for t in range(1,10)],1)
  for a in range(1,10):
    for b in range(a+1,10): F[f'gW[{a}:{b}]']=g[:,a:b+1].sum(1)
  return F
def pool4(x):
  F=pool3(x); s=-np.sort(-x,1)
  for k in range(10):
    F[f'relu(x{k}-3rd)']=np.maximum(x[:,k]-s[:,2],0)
    F[f'T*[x{k} top2]']=s[:,:2].mean(1)*(x[:,k]>=s[:,1])
    F[f'top-x{k}']=s[:,0]-x[:,k]; F[f'sec-x{k}']=s[:,1]-x[:,k]
    if k>=2:
      p=-np.sort(-x[:,:k],1)
      F[f'relu(x{k}-2ndprev)']=np.maximum(x[:,k]-p[:,1],0)
  return F
