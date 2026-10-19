from lib import *
import sys
n=20000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
L={}
for t in range(10): L[f'x{t}']=X[:,t]; L[f'|x{t}|']=abs(X[:,t])
for t in range(1,10):
    L[f'G{t}']=np.maximum(Mm[:,t-1]-X[:,t],0); L[f'M{t}']=Mm[:,t]; L[f'S{t}']=Ss[:,t]
    if t>=2: L[f'g{t}']=np.maximum(Ss[:,t-1]-X[:,t],0)
for s in range(10):
    for t in range(s+1,10): L[f'D{s}{t}']=np.maximum(X[:,s]-X[:,t],0)
L['EM']=Mm[:,1:].mean(1)*10
names=list(L); A=np.c_[[L[k] for k in names]].T
for i in [int(a) for a in sys.argv[1:]]:
    y=Hs[:,i]; sel=[]
    for step in range(5):
        best=None
        for j,nm in enumerate(names):
            if j in sel: continue
            R,c,p=relufit(A[:,sel+[j]],y,it=5)
            if best is None or R>best[0]: best=(R,j,c)
        sel.append(best[1]); print(i,step,[names[s] for s in sel],round(best[0],4),best[2].round(2),flush=True)
