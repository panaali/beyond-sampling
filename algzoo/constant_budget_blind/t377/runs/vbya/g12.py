import numpy as np, feats, probe
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(17)
x=rng.standard_normal((30000,10)); H=probe.hidden(x)
R=lambda z:np.maximum(z,0)
pre=x[:,:9]; q=-np.sort(-pre,1); am=pre.argmax(1); gap=q[:,0]-q[:,1]
def r2(i,A,relu=True):
    c=feats.relufit(H[:,i],A) if relu else np.linalg.lstsq(A,H[:,i],rcond=None)[0]
    p=R(A@c) if relu else A@c
    return 1-((H[:,i]-p)**2).mean()/H[:,i].var(),c
one=np.ones(len(x))
for i in [15,0,4,6]:
    best=None
    for rr in np.linspace(0.3,1.6,27):
        wgap=gap*rr**am
        for nm,A in [('wgap,ps1,ps2',np.c_[wgap,q[:,0],q[:,1],one]),('wgap,ps1,ps2,ps3',np.c_[wgap,q[:,0],q[:,1],q[:,2],one])]:
            r,c=r2(i,A)
            if best is None or r>best[0]: best=(r,rr,nm,c)
    print(i,'decay model best R2 %.3f r=%.2f %s'%best[:3],best[3])
    # per-argmax-position weights upper bound: gap*[am=k] for each k + ps1,ps2,ps3
    A=np.c_[np.stack([gap*(am==k) for k in range(9)],1),q[:,:3],one]
    r,c=r2(i,A); print('   per-pos gap weights R2 %.3f'%r,c)
    A=np.c_[np.stack([gap*(am==k) for k in range(9)],1),np.stack([(am==k)*1. for k in range(9)],1),q[:,:3]]
    r,c=r2(i,A); print('   per-pos gap+offset R2 %.3f'%r,c)
