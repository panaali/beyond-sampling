from lib import *
import sys
n=20000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
L={}
for t in range(10): L[f'x{t}']=X[:,t]
for t in range(1,10):
    L[f'G{t}']=np.maximum(Mm[:,t-1]-X[:,t],0)
    if t>=2: L[f'g{t}']=np.maximum(Ss[:,t-1]-X[:,t],0)
L['M8']=Mm[:,8]; L['S8']=Ss[:,8]
for lam in [1.0,1.2,1.4,1.7]:
    wts=lam**(9-np.arange(10)); wts/=wts.sum()
    L[f'EM{lam}']=(Mm[:,:]*wts)[:,1:].sum(1)*10
    L[f'ES{lam}']=(Ss[:,1:]*wts[1:]).sum(1)*10
    L[f'EX{lam}']=(X*wts).sum(1)*10
names=list(L); A=np.c_[[L[k] for k in names]].T
for i in [int(a) for a in sys.argv[1:]]:
    y=Hs[:,i]; sel=[]
    for step in range(5):
        best=None
        for j,nm in enumerate(names):
            if j in sel: continue
            R,c,p=relufit(A[:,sel+[j]],y,it=6)
            if best is None or R>best[0]: best=(R,j,c)
        sel.append(best[1]); print(i,step,[names[s] for s in sel],round(best[0],4),best[2].round(2),flush=True)
