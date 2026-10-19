from lib import *
import sys, itertools
n=10000; X=x[:n]; Hs=H[:n]; Mm=M[:n]; Ss=S[:n]
def topmean(a,k=3):
    s=-np.sort(-a,1); k=min(k,a.shape[1]); return s[:,:k].mean(1)
lev={'EM':Mm[:,1:].mean(1)*10,'T3all':topmean(X),'T3pre':topmean(X[:,:9])}
T={}
for t in range(10): T[f'x{t}']=X[:,t]
for t in range(1,10): T[f'gap{t}']=np.maximum(topmean(X[:,:t])-X[:,t],0)
tn=list(T); TA=np.c_[[T[k] for k in tn]].T
E={}
for t in range(1,10): E[f'G{t}']=np.maximum(Mm[:,t-1]-X[:,t],0)
for t in range(2,10): E[f'g{t}']=np.maximum(Ss[:,t-1]-X[:,t],0)
for s in range(10):
    for t in range(s+1,10): E[f'D{s}{t}']=np.maximum(X[:,s]-X[:,t],0)
for t in range(10): E[f'|x{t}|']=abs(X[:,t])
E.update(T); en=list(E); EA=np.c_[[E[k] for k in en]].T
for i in [int(a) for a in sys.argv[1:]]:
    y=Hs[:,i]; res=[]
    for ln,lv in lev.items():
        for comb in itertools.combinations(range(len(tn)),3):
            R,c,p=relufit(np.c_[lv,TA[:,list(comb)]],y,it=4)
            res.append((R,ln,list(comb)))
    res.sort(key=lambda a:-a[0])
    for r in res[:3]:
        base=np.c_[lev[r[1]],TA[:,r[2]]]
        best=max(((relufit(np.c_[base,EA[:,j]],y,it=4)[0],j) for j in range(len(en))))
        R5,c5,_=relufit(np.c_[base,EA[:,best[1]]],y)
        print(i,round(r[0],4),r[1],[tn[k] for k in r[2]],'| +',en[best[1]],round(R5,4),c5.round(3),flush=True)
