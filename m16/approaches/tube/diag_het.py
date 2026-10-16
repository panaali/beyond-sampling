import sys
sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/tube')
sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
n=16384
for fn,name,ncomb in [(faces23,'23',360),(faces12,'12',45)]:
    f=fn(n); F,_,C=F_faces(M,f,400,radial=True); F*=f['Z']
    np.savez(f'het{name}.npz',F=F,C=C,x0=f['x0'],tpos=f['tpos'],tneg=f['tneg'])
    key=f['tpos']*100+f['tneg'] if name=='12' else None
    if name=='23':
        cidx=np.array([COMBO23.index(tuple(r)) for r in np.stack([f['tpos'],f['tneg'],np.argmax(np.where(np.isin(np.arange(10)[None],np.stack([f['tpos'],f['tneg']],1)[:,:,None].any(1)) ,-9,f['x0']),1)],1)]) if False else None
    # group key: positions (a,b) + c (argmax of x0)
    c=np.argmax(f['x0'],1); a=f['tpos']; b=f['tneg']
    key=(a*10+b)*10+(c if name=='23' else 0)
    ks=np.unique(key); sd=np.array([F[key==k].std() for k in ks]); cs=np.array([C[key==k].mean() for k in ks]); w=np.array([(key==k).mean() for k in ks])
    tot=F.var()*C.mean()
    within=(w*sd**2).sum()*C.mean()
    neyman=(w*sd*np.sqrt(cs)).sum()**2
    print(name,'var*cost total',tot,'stratified',within,'neyman',neyman, 'mean',F.mean())
