import sys,numpy as np; sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/cheap_frontier'); import cf
ms=cf.lb3_models(); sel={'orig':ms['orig'],'seed0':ms['seed0'],'seed3':ms['seed3']}
ts=[0.5,0.75,1.0,1.25,1.5,2.0]
for nm,(Wi,Wh,Wo) in sel.items():
    rng=np.random.default_rng(5); N=0; cnt={}
    for _ in range(40):
        x=rng.standard_normal((500_000,10)); h=np.zeros((len(x),16))
        for t in range(10): h=np.maximum(h@Wh.T+np.outer(x[:,t],Wi),0)
        yhat=(h@Wo.T).argmax(1); o=np.argsort(-x,1); r=np.arange(len(x))
        v=-np.sort(-x,1); g12=v[:,0]-v[:,1]; g23=v[:,1]-v[:,2]
        rank=np.argmax(o==yhat[:,None],1); wrong=rank!=1; blind=v[:,1]<=0
        for t in ts:
            for lab,m in (("any",wrong),("rank3",rank==2),("rank1",rank==0),("rank4+",rank>=3)):
                k=(lab,t); cnt[k]=cnt.get(k,0)+int((m&(g23>t)).sum())
            k=('any_g12>.5',t); cnt[k]=cnt.get(k,0)+int((wrong&(g23>t)&(g12>.5)).sum())
            k=('any_nonblind',t); cnt[k]=cnt.get(k,0)+int((wrong&(g23>t)&~blind).sum())
            k=('P(g23>t)',t); cnt[k]=cnt.get(k,0)+int((g23>t).sum())
        N+=len(x)
    print(nm,'N',N)
    for lab in ("P(g23>t)","any","rank3","rank1","rank4+","any_g12>.5","any_nonblind"):
        print('  %-14s'%lab," ".join("t%.2f:%.1e"%(t,cnt[(lab,t)]/N) for t in ts))
