import numpy as np, probe
rng=np.random.default_rng(5)
x=rng.standard_normal((100000,10))
H=probe.hidden(x); np.save('x.npy',x); np.save('H.npy',H)
p=probe.predict(x)
srt=np.argsort(-x,1); sec=srt[:,1]; am=srt[:,0]
print('acc',(p==sec).mean())
print('mean',H.mean(0).round(3)); print('std',H.std(0).round(3)); print('frac0',(H<1e-6).mean(0).round(3))
# correlations with features
feats={}
for t in range(10): feats[f'x{t}']=x[:,t]
sx=-np.sort(-x,1)
feats['max']=sx[:,0]; feats['2nd']=sx[:,1]; feats['gap']=sx[:,0]-sx[:,1]
for t in range(10): feats[f'am{t}']=(am==t)*1.; 
for t in range(10): feats[f'sec{t}']=(sec==t)*1.
for i in range(8):
    c={k:np.corrcoef(v,H[:,i])[0,1] for k,v in feats.items()}
    top=sorted(c.items(),key=lambda a:-abs(a[1]))[:8]
    print(i,[(k,round(v,2)) for k,v in top])
