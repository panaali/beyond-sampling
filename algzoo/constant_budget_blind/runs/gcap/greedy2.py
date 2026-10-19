from lib import *
import sys
n=20000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
sx=-np.sort(-X,1)
L={}
for t in range(10): L[f'x{t}']=X[:,t]; L[f'p{t}']=np.maximum(X[:,t],0); L[f'n{t}']=np.maximum(-X[:,t],0)
for t in range(1,10): L[f'M{t}']=Mm[:,t]; L[f'S{t}']=Ss[:,t]
for k in range(4): L[f'top{k}']=sx[:,k]
for t in range(1,10):
    L[f'G{t}']=np.maximum(Mm[:,t-1]-X[:,t],0); L[f'U{t}']=np.maximum(X[:,t]-Mm[:,t-1],0)
    if t>=2: L[f'g{t}']=np.maximum(Ss[:,t-1]-X[:,t],0); L[f'u{t}']=np.maximum(X[:,t]-Ss[:,t-1],0)
L['sum']=X.sum(1); L['abs']=np.abs(X).sum(1)
for t in range(1,10): L[f'sum{t}']=X[:,:t+1].sum(1)
names=list(L); A=np.c_[[L[k] for k in names]].T
neur=[int(a) for a in sys.argv[1:]] or range(8)
for i in neur:
    y=Hs[:,i]; sel=[]
    for step in range(5):
        best=None
        for j,nm in enumerate(names):
            if j in sel: continue
            R,c,p=relufit(A[:,sel+[j]],y,it=6)
            if best is None or R>best[0]: best=(R,j,c)
        sel.append(best[1]); print(i,step,[names[s] for s in sel],round(best[0],4),best[2].round(2),flush=True)
