import numpy as np, probe, sys
from gate import fastfit, lmfit
rng=np.random.default_rng(41)
X0=rng.standard_normal((20000,10)); H=probe.hidden(X0)
Xt0=np.random.default_rng(42).standard_normal((20000,10)); Ht=probe.hidden(Xt0)
def base(x):
    p=x[:,:9]; s=np.argsort(-p,1); srt=-np.sort(-p,1)
    return s,srt
def feats(x,om,o2,o3):
    s,srt=base(x); F={}
    F['M9']=srt[:,0];F['S9']=srt[:,1];F['P3']=srt[:,2];F['SM9']=x[:,:9].sum(1);F['x9']=x[:,9]
    for k in range(10): F['x%d'%k]=x[:,k]
    for nm,o,col in [('M9',om,0),('S9',o2,1),('P3',o3,2)]:
        for j in range(1,9):
            for rev in [0,1]:
                S=o[:j] if not rev else o[-j:]
                F['%s*[pos in %s]'%(nm,''.join(map(str,sorted(S))))]=srt[:,col]*np.isin(s[:,col],S)
    return F
for i in [int(a) for a in sys.argv[1:]]:
    h=H[:,i]; s,srt=base(X0); E=np.eye(9)
    X=np.c_[srt[:,0:1]*E[s[:,0]], srt[:,1:2]*E[s[:,1]], srt[:,2:3]*E[s[:,2]], X0[:,:9].sum(1), X0[:,9]]
    c,_=fastfit(h,X)
    om=list(np.argsort(-c[:9])); o2=list(np.argsort(-c[9:18])); o3=list(np.argsort(-c[18:27]))
    F=feats(X0,om,o2,o3); Ft=feats(Xt0,om,o2,o3); names=list(F); sel=[]
    print(i,flush=True)
    for step in range(5):
        best=None
        for n in names:
            if n in sel: continue
            A=np.stack([F[q] for q in sel+[n]],1); cc,r=fastfit(h,A)
            if best is None or r>best[1]: best=(n,r)
        sel.append(best[0]); A=np.stack([F[q] for q in sel],1); cc,_=fastfit(h,A); cc=lmfit(h,A,cc)
        At=np.stack([Ft[q] for q in sel],1); r2t=1-((np.maximum(At@cc,0)-Ht[:,i])**2).mean()/Ht[:,i].var()
        print("   %.3f "%r2t+" + ".join("%.2f*%s"%(a,b) for a,b in zip(cc,sel)),flush=True)
