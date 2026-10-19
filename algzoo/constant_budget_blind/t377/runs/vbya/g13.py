import numpy as np, feats, probe, json, sys
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(18)
x=rng.standard_normal((20000,10)); H=probe.hidden(x)
xt=rng.standard_normal((30000,10)); Ht=probe.hidden(xt)
R=lambda z:np.maximum(z,0)
def base(x):
    pre=x[:,:9]; q=-np.sort(-pre,1); am=pre.argmax(1); return q,am,q[:,0]-q[:,1]
q,am,gap=base(x); qt,amt,gapt=base(xt)
def A(lab,q,am,gap):
    return np.c_[gap*(lab[am]==1),gap*(lab[am]==2),q[:,0],q[:,1],np.ones(len(q))]
def sc(i,lab):
    c=feats.relufit(H[:,i],A(lab,q,am,gap)); p=R(A(lab,q,am,gap)@c); return 1-((H[:,i]-p)**2).mean()/H[:,i].var(),c
out={}
for i in [int(a) for a in sys.argv[1:]]:
    best=None
    for init in range(3):
        lab=np.zeros(9,int) if init==0 else rng.integers(0,3,9)
        cur=sc(i,lab)[0]; imp=True
        while imp:
            imp=False
            for k in range(9):
                for v in range(3):
                    if v==lab[k]: continue
                    l2=lab.copy(); l2[k]=v; r=sc(i,l2)[0]
                    if r>cur+1e-4: cur=r; lab=l2; imp=True
        if best is None or cur>best[0]: best=(cur,lab.copy())
    lab=best[1]; r,c=sc(i,lab); p=R(A(lab,qt,amt,gapt)@c); rt=1-((Ht[:,i]-p)**2).mean()/Ht[:,i].var()
    T1=[k for k in range(9) if lab[k]==1]; T2=[k for k in range(9) if lab[k]==2]
    print(i,'T1',T1,'T2',T2,'coef(G1,G2,ps1,ps2,1)',c,'train %.3f test %.3f'%(r,rt))
    out[i]=dict(T1=T1,T2=T2,c=[float(v) for v in c],r2=float(rt))
json.dump(out,open('fit_gap2_%s.json'%'_'.join(sys.argv[1:]),'w'))
