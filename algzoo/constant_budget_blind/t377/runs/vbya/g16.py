import numpy as np, probe, feats
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(22)
x=rng.standard_normal((20000,10)); H=probe.hidden(x)
xt=rng.standard_normal((30000,10)); Ht=probe.hidden(xt)
R=lambda z:np.maximum(z,0)
def B(x):
    p=x[:,:9]; s=-np.sort(-p,1); am=p.argmax(1)
    po2_0=-np.sort(-p[:,1:],1)[:,1]
    po=lambda t,k: -np.sort(-np.delete(p,t,1),1)[:,k-1]
    return dict(s=s,am=am,gap=s[:,0]-s[:,1],E0=R(x[:,0]-po2_0),x5=x[:,5],E5=R(x[:,5]-po(5,2)),D5=R(x[:,5]-po(5,1)),R5=R(po(5,1)-x[:,5]))
b=B(x); bt=B(xt)
def run(i,mk,lab):
    A=mk(b); At=mk(bt); c=feats.relufit(H[:,i],A); p=R(At@c)
    r=1-((Ht[:,i]-p)**2).mean()/Ht[:,i].var(); return r,c
one=lambda d: np.ones(len(d['am']))
best=None
for rr in [0.4,0.5,0.6,0.7,0.8,0.9]:
    for nm,mk in [('wg,E0,ps1,ps2',lambda d,rr=rr: np.c_[d['gap']*rr**d['am'],d['E0'],d['s'][:,0],d['s'][:,1],one(d)]),
                  ('wg,E0,ps2,ps3',lambda d,rr=rr: np.c_[d['gap']*rr**d['am'],d['E0'],d['s'][:,1],d['s'][:,2],one(d)])]:
        r,c=run(0,mk,nm)
        if best is None or r>best[0]: best=(r,rr,nm,c)
print('neuron0',best)
# neuron 12: search set S for G_S, with E5/R5 features
def mk12(S,var):
    def f(d):
        G=np.isin(d['am'],S)*d['gap']
        if var==0: return np.c_[G,d['R5'],d['E5'],d['s'][:,1],one(d)]
        if var==1: return np.c_[G,np.isin(d['am'],S)*1.,d['E5'],d['s'][:,0],one(d)]
        return np.c_[G,d['E5'],d['D5'],d['s'][:,1]-d['s'][:,2],one(d)]
    return f
for var in [0,1,2]:
    S=[2,3,4,6,7]; cur=run(12,mk12(S,var),'')[0]; imp=True
    while imp:
        imp=False
        for k in range(9):
            S2=[u for u in S if u!=k] if k in S else sorted(S+[k])
            r=run(12,mk12(S2,var),'')[0]
            if r>cur+1e-3: cur=r;S=S2;imp=True
    print('neuron12 var',var,S,run(12,mk12(S,var),''))
