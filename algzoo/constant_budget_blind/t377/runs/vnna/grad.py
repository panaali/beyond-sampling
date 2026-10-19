import numpy as np, probe, sys
np.set_printoptions(precision=2,suppress=True,linewidth=200)
rng=np.random.default_rng(14)
N=20000; x=rng.standard_normal((N,10)); H=probe.hidden(x); eps=1e-3
G=np.zeros((N,10,32))
for t in range(10):
    xp=x.copy(); xp[:,t]+=eps; G[:,t,:]=(probe.hidden(xp)-H)/eps
np.save('G.npy',G[:5000]); np.save('Gx.npy',x[:5000])
s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
for i in [int(a) for a in sys.argv[1:]]:
    g=G[:,:,i]; act=H[:,i]>0
    print(i,"mean grad (active)",g[act].mean(0))
    # mean grad conditional: at argmax position, at a2 position, elsewhere
    ga=g[np.arange(N),am][act].mean(); g2=g[np.arange(N),a2][act].mean()
    print("   grad at am %.2f at a2 %.2f; fraction samples where top-|grad| pos == am %.2f, == a2 %.2f"%(ga,g2,(np.abs(g).argmax(1)==am)[act].mean(),(np.abs(g).argmax(1)==a2)[act].mean()))
    # check homogeneity: sum g*x vs h
    print("   Euler check corr", np.corrcoef((g*x).sum(1),H[:,i])[0,1])
