import numpy as np, feats, probe, json
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(15)
x=rng.standard_normal((12000,10)); H=probe.hidden(x)
xt=rng.standard_normal((20000,10)); Ht=probe.hidden(xt)
F,u9=feats.build2(x); Ft,_=feats.build2(xt)
names=[n for n in F if not u9[n]]
R=lambda z:np.maximum(z,0)
pam=x[:,:9].argmax(1); pamt=xt[:,:9].argmax(1)
out={}
for i in [0,4,5,6,12,15,29,31,10]:
    res=[]
    for gate in [None]+list(range(9)):
        for gmode in (['excl','only'] if gate is not None else ['none']):
            if gate is None: m=np.ones(len(x),bool); mt=np.ones(len(xt),bool)
            elif gmode=='excl': m=pam!=gate; mt=pamt!=gate
            else: m=pam==gate; mt=pamt==gate
            y=H[m,i]
            if y.var()<1e-9: continue
            Fs={n:F[n][m] for n in names}
            hist=feats.greedy_act(y,Fs,names,3)
            r,sel,c=hist[-1]
            At=np.c_[np.stack([Ft[s] for s in sel],1),np.ones(len(xt))]
            p=R(At@c)*mt
            rt=1-((Ht[:,i]-p)**2).mean()/Ht[:,i].var()
            res.append((rt,gate,gmode,sel,c))
    res.sort(key=lambda z:-z[0])
    for rt,gate,gmode,sel,c in res[:3]:
        print(i,'test R2 %.3f gate %s %s'%(rt,gate,gmode),sel,c)
    rt,gate,gmode,sel,c=res[0]
    out[i]=dict(gate=gate,gmode=gmode,sel=sel,c=[float(v) for v in c],r2=float(rt))
json.dump(out,open('fit_gate.json','w'))
