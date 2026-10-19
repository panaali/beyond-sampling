import numpy as np, json
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:20000].astype(float); h=d["h"][:20000].astype(float)
def oth(k): return np.sort(np.delete(x,k,1),1)
def below(k): return np.maximum(oth(k)[:,-1]-x[:,k],0)
def qgap(Q,R):
    rest=[j for j in R if j not in Q]
    if not Q or not rest: return np.zeros(len(x))
    return np.maximum(x[:,Q].max(1)-x[:,rest].max(1),0)
res={}
am=np.argmax(x,1)
for i,k in [(3,9),(9,9),(17,9),(18,9),(20,9),(22,9),(28,9),(30,9),(14,8),(24,7),(12,5)]:
    y=h[:,i]; R=[j for j in range(10) if j!=k]; b=below(k)
    mm={j:y[am==j].mean() for j in R}; med=np.median(list(mm.values()))
    def sc(st):
        Qn=[j for j in R if st[j]==1]; Qp=[j for j in R if st[j]==2]
        return relufit(np.c_[b,qgap(Qn,R),qgap(Qp,R)],y,iters=10)
    st={j:(2 if mm[j]>med*1.15 else (1 if mm[j]<med*0.85 else 0)) for j in R}
    c,r=sc(st); imp=True
    while imp:
        imp=False
        for j in R:
            for s in (0,1,2):
                if s==st[j]: continue
                st2=dict(st); st2[j]=s; c2,r2_=sc(st2)
                if r2_>r+1e-4: st,c,r=st2,c2,r2_; imp=True
    Qn=[j for j in R if st[j]==1]; Qp=[j for j in R if st[j]==2]
    res[i]=dict(k=k,Qn=Qn,Qp=Qp)
    print("X9",i,"k",k,"Qn",Qn,"Qp",Qp,"R2 %.3f"%r,np.round(c,2),flush=True)
json.dump(res,open("s17.json","w"))
for i,k in [(1,2),(2,4),(11,3),(21,1),(23,6),(26,7),(29,0),(10,8),(31,8),(5,0),(12,5)]:
    y=h[:,i]; o=oth(k); xk=x[:,k]
    F={"isMax":np.maximum(xk-o[:,-1],0),"inTop2":np.maximum(xk-o[:,-2],0),"inTop3":np.maximum(xk-o[:,-3],0),
       "below":np.maximum(o[:,-1]-xk,0),"belowSec":np.maximum(o[:,-2]-xk,0),"belowThird":np.maximum(o[:,-3]-xk,0),
       "xk":xk,"S0":-np.sort(-x,1)[:,0]}
    for combo in [["isMax","inTop2","inTop3"],["isMax","inTop2","inTop3","S0"],["below","belowSec","belowThird"],["below","belowSec","S0"],["isMax","inTop2","xk","S0"]]:
        c,r=relufit(np.stack([F[n] for n in combo],1),y,iters=15)
        print("DET",i,k,combo,"%.3f"%r,np.round(c,2))
