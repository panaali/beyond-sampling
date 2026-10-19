import numpy as np, feats, probe
np.set_printoptions(precision=2,suppress=True,linewidth=220)
rng=np.random.default_rng(7)
x=rng.standard_normal((20000,10))
H=probe.hidden(x)
F,u9=feats.build(x); names=[n for n in F if not u9[n]]
res={}
for sgn,idx in [(-1,[3,7,9,17,18,20,22,28,30]),(1,[8,13,19])]:
    y=x.copy(); y[:,9]=20*sgn; ha=probe.hidden(y); y[:,9]=25*sgn; hb=probe.hidden(y)
    for i in idx:
        w=np.median((hb[:,i]-ha[:,i])/(5*sgn)); g=ha[:,i]-w*20*sgn
        ok=ha[:,i]>0
        hist=feats.greedy(g[ok],{n:F[n][ok] for n in names},names,4)
        print(i,'w=%.3f ok %.2f'%(w,ok.mean()),' | '.join('%.3f %s'%(r,s[-1]) for r,s,c in hist),' coef',hist[-1][2])
        for k in [1,2,3]:
            r,s,c=hist[k]
            gh=np.stack([F[n] for n in s],1)@c[:-1]+c[-1]
            hp=np.maximum(w*x[:,9]+gh,0)
            print('    k=%d h-R2 %.3f'%(k+1,1-((H[:,i]-hp)**2).mean()/H[:,i].var()))
