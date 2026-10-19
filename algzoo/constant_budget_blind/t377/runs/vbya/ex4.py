import numpy as np, probe
np.set_printoptions(precision=2,suppress=True,linewidth=200)
rng=np.random.default_rng(3)
x=rng.standard_normal((50000,10))
y=x.copy(); y[:,9]=-20; ha=probe.hidden(y)
y[:,9]=-25; hb=probe.hidden(y)
w=(ha-hb)/5
wm=np.median(w,0)
G=ha+20*wm
np.save('x4.npy',x); np.save('G4.npy',G); np.save('w4.npy',wm)
pre=x[:,:9]; P8=pre.max(1); s=-np.sort(-pre,1); am=pre.argmax(1); a2=np.argsort(-pre,1)[:,1]
for i in [3,7,9,17,18,20,22,28,30,25]:
    g=G[:,i]
    ok=np.abs(w[:,i]-wm[i])<1e-3
    F=np.c_[pre,P8,s[:,1]]
    A=np.c_[F[ok],np.ones(ok.sum())]
    c,*_=np.linalg.lstsq(A,g[ok],rcond=None); r=g[ok]-A@c
    print(i,'w=%.2f okfrac %.3f'%(wm[i],ok.mean()),'coef x0..8,P8,S2,1:',c,'R2 %.3f'%(1-r.var()/g[ok].var()))
    print('   mean g | am:',np.array([g[ok&(am==t)].mean() - wm[i]*0 for t in range(9)]), ' g-|w|P8 |am', np.array([(g+wm[i]*P8)[ok&(am==t)].mean() for t in range(9)]))
