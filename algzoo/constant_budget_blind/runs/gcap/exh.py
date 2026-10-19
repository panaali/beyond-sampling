from lib import *
import sys, itertools
n=10000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
def top(a,k): return -np.sort(-a,1)[:,k]
lev={'EM':Mm[:,1:].mean(1)*10,'M9':Mm[:,9],'top3':top(X,0)+top(X,1)+top(X,2)}
T={}
for t in range(10): T[f'x{t}']=X[:,t]
for t in range(1,10): T[f'G{t}']=np.maximum(Mm[:,t-1]-X[:,t],0)
for t in range(2,10): T[f'g{t}']=np.maximum(Ss[:,t-1]-X[:,t],0)
tn=list(T); TA=np.c_[[T[k] for k in tn]].T
for i in [int(a) for a in sys.argv[1:]]:
    y=Hs[:,i]; res=[]
    for ln,lv in lev.items():
        for comb in itertools.combinations(range(len(tn)),3):
            R,c,p=relufit(np.c_[lv,TA[:,list(comb)]],y,it=4)
            res.append((R,ln,[tn[k] for k in comb],c.round(2)))
    res.sort(key=lambda a:-a[0])
    for r in res[:4]: print(i,round(r[0],4),r[1],r[2],r[3],flush=True)
