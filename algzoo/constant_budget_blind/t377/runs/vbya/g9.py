import numpy as np, feats, probe, json, itertools
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(14)
x=rng.standard_normal((20000,10)); H=probe.hidden(x)
xt=rng.standard_normal((20000,10)); Ht=probe.hidden(xt)
F,_=feats.build2(x); Ft,_=feats.build2(xt)
R=lambda z:np.maximum(z,0)
def ev(i,sel,relu):
    A=np.c_[np.stack([F[s] for s in sel],1),np.ones(len(x))]
    At=np.c_[np.stack([Ft[s] for s in sel],1),np.ones(len(xt))]
    if relu: c=feats.relufit(H[:,i],A); p=R(At@c)
    else: c,*_=np.linalg.lstsq(A,H[:,i],rcond=None); p=At@c
    return 1-((Ht[:,i]-p)**2).mean()/Ht[:,i].var(),c
out={}
for i,t in [(21,1),(1,2),(11,3),(2,4),(29,0),(23,6),(26,7)]:
    pool=['R(x%d-po1)'%t,'R(x%d-po2)'%t,'R(x%d-o1)'%t,'R(x%d-o2)'%t,'R(x%d-o3)'%t,'[pam=%d]'%t,'[pa2=%d]'%t,'R(x0-po1)','x%d'%t]
    if t>=1: pool+=['R(x%d-P%d)'%(t,t-1),'R(P%d-x%d)'%(t-1,t)]
    if t>=2: pool+=['R(x%d-Q2_%d)'%(t,t),'R(Q2_%d-x%d)'%(t,t)]
    pool=list(dict.fromkeys(pool))
    best=None
    for k in [2,3,4]:
        for sel in itertools.combinations(pool,k):
            for relu in [True]:
                r,c=ev(i,list(sel),relu)
                if best is None or r>best[0]+ (0.003 if k>len(best[1]) else 0): best=(r,list(sel),relu,c)
    print(i,'R2test %.3f'%best[0],best[1],best[3])
    out[i]=dict(sel=best[1],relu=best[2],c=[float(v) for v in best[3]],r2=float(best[0]))
json.dump(out,open('fit_top.json','w'))
