import numpy as np,sys
from relufit import relufit
d=np.load('d1.npz'); x,h=d['x'][:30000],d['h'][:30000]
p=x[:,:9]; Q=-np.sort(-p,1); x9=x[:,9]
fam={}
fam['Qs']=np.c_[Q[:,0],Q[:,1],Q[:,2],Q[:,3],x9]
for j in range(9):
    o=-np.sort(-np.delete(p,j,1),1)
    fam[f'O{j}']=np.c_[p[:,j],o[:,0],o[:,1],o[:,2],x9]
    fam[f'Q{j}']=np.c_[p[:,j],Q[:,0],Q[:,1],Q[:,2],x9]
for i in map(int,sys.argv[1:]):
    res=[]
    for k,A in fam.items():
        co,R=relufit(A,h[:,i]); res.append((R,k,co))
    res.sort(key=lambda t:-t[0])
    print(i,' | '.join(f'{k}:{R:.3f} {np.round(co,2)}' for R,k,co in res[:3]))
