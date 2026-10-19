import numpy as np, probe, labels
x=np.random.default_rng(777).standard_normal((50000,10)); H=probe.hidden(x)
r=[]
for i in range(32):
    p=labels.f(i,x); y=H[:,i]; r.append(1-((p-y)**2).mean()/y.var())
print(' '.join('%d:%.3f'%(i,v) for i,v in enumerate(r))); print('mean %.3f'%np.mean(r))
# readout check: does a linear readout of predicted h reproduce answers?
P=np.stack([labels.f(i,x) for i in range(32)],1)
print([labels.f(i,np.random.default_rng(1).standard_normal((1000,10))).shape for i in range(32)][:3], all(len(labels.LABELS[i][1].split())<=40 for i in range(32)), max(labels.CONSTANTS.values()))
y=probe.predict(x); Y=np.eye(10)[y]; A=np.c_[H,np.ones(len(H))]
Wt,*_=np.linalg.lstsq(A[:25000],Y[:25000],rcond=None)
print('linear-probe acc on true h %.3f, on formula h %.3f'%(((A[25000:]@Wt).argmax(1)==y[25000:]).mean(),((np.c_[P,np.ones(len(P))][25000:]@Wt).argmax(1)==y[25000:]).mean()))
