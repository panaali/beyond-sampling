# diagnostic: closure h_10 moments and logit-margin P vs MC for a few units of one model
import sys, os; sys.argv=['x']; os.chdir('/home/user/clean-neurons')
sys.path[:0]=['m16/approaches/meanfield','m16/approaches/blind11','m16']
import numpy as np, mf2, blind as B11, leaderboard as lb
z=np.load('/tmp/claude-0/dev2/weights.npz'); i=int(os.environ.get('MI',0))
m=(z['Wi'][i].astype(float),z['Wh'][i].astype(float),z['Wo'][i].astype(float)); Wi,Wh,Wo=m
rng=np.random.default_rng(5); T=10
def sample(case,a,j,n=400000):
    if case is None:
        Zs=-np.sort(np.abs(rng.standard_normal((n,10))),1); X=np.empty((n,10)); X[:,a]=Zs[:,0]; X[:,j]=Zs[:,1]
        rest=[p for p in range(10) if p not in (a,j)]; perm=np.argsort(rng.random((n,8)),1)
        X[:,rest]=np.take_along_axis(Zs[:,2:],perm,1)
    else:
        Zs=-np.sort(np.abs(rng.standard_normal((n,9))),1); X=np.empty((n,10)); X[:,case]=np.abs(rng.standard_normal(n)); X[:,j]=Zs[:,0]
        rest=[p for p in range(10) if p not in (case,j)]; perm=np.argsort(rng.random((n,8)),1)
        X[:,rest]=np.take_along_axis(Zs[:,1:],perm,1)
    return X
for (case,a,j) in [(None,0,1),(None,7,3),(3,None,5),(9,None,0),(0,None,8)]:
    X=sample(case,a,j); h=np.zeros((len(X),16))
    for t in range(T): h=np.maximum(h@Wh.T+X[:,t:t+1]*Wi[None],0)
    L=h@Wo.T; pc=(L.argmax(1)==j).mean(); mLs=L.mean(0); CLs=np.cov(L.T)
    row=[f"case {case},{a},{j} MC P={pc:.4f}"]
    for v in "0abc":
        ops=mf2.Ops(); mu_x,S_x=B11.moments(case,a,j)
        mL,CL=mf2.propagate(m,mu_x,S_x,mf2.laws_for(case,a,j) if v in "bc" else None, v in "ac", ops)
        Bm=np.eye(10)[j][None,:]-np.delete(np.eye(10),j,0); o=B11.Ops(); o.n=0
        p=B11.clark_p_all_nonneg(Bm@mL,Bm@CL@Bm.T,o)
        # also clark on the MC Gaussian moments (isolates the readout closure)
        po=B11.clark_p_all_nonneg(Bm@mLs,Bm@CLs@Bm.T,o)
        row.append(f"{v}:P={p:.4f} dmean={np.abs(mL-mLs).max():.2f} dsd={np.abs(np.sqrt(np.diag(CL))-np.sqrt(np.diag(CLs))).max():.2f}")
    row.append(f"clark(MC moments)={po:.4f}")
    print(" | ".join(row))
print("--- readout split: MC truth | Gaussian(MC moments) exact orthant (MC 1e6) | Clark(MC moments)")
for (case,a,j) in [(None,0,1),(None,7,3),(3,None,5),(9,None,0),(0,None,8)]:
    X=sample(case,a,j); h=np.zeros((len(X),16))
    for t in range(T): h=np.maximum(h@Wh.T+X[:,t:t+1]*Wi[None],0)
    L=h@Wo.T; Bm=np.eye(10)[j][None,:]-np.delete(np.eye(10),j,0); M=L@Bm.T
    mm,CC=M.mean(0),np.cov(M.T); G=rng.multivariate_normal(mm,CC,1000000)
    o=B11.Ops(); o.n=0
    print(case,a,j,"%.4f | %.4f | %.4f"%((M>=0).all(1).mean(),(G>=0).all(1).mean(),B11.clark_p_all_nonneg(mm,CC,o)))
