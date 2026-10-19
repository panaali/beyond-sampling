import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
H=probe.hidden(x); H=np.asarray(H)
np.save('x1.npy',x); np.save('H1.npy',H)
pred=np.asarray(probe.predict(x))
srt=np.argsort(-x,1); am=srt[:,0]; a2=srt[:,1]
print('acc',(pred==a2).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
def corr(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    return (a.T@b)/np.sqrt((a**2).sum(0)[:,None]*(b**2).sum(0)[None,:]+1e-12)
OH1=np.eye(10)[am];OH2=np.eye(10)[a2]
for i in range(32):
    h=H[:,i:i+1]
    print(i,'act%.2f mean%.2f sd%.2f'%((h>1e-6).mean(),h.mean(),h.std()))
    print('  x  ',corr(h,x)[0]); print('  am ',corr(h,OH1)[0]); print('  a2 ',corr(h,OH2)[0])
