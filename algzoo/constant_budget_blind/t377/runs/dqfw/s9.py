import numpy as np, probe
rng=np.random.default_rng(2)
x=rng.standard_normal((40000,10))
for T in [10,9,8,7]:
    k=10-T
    xp=np.concatenate([np.zeros((len(x),k)),x[:,:T]],1)
    h=probe.hidden(xp)
    xs=x[:,:T]; am=np.argmax(xs,1)
    out=[]
    for i in range(32):
        c=[np.corrcoef(h[:,i],am==T-1-l)[0,1] if h[:,i].std()>0 else 0 for l in range(T)]
        l=int(np.nanargmax(np.abs(c))); out.append('%d:%d(%+.2f)'%(i,l,c[l]))
    print('T',T,' '.join(out))
