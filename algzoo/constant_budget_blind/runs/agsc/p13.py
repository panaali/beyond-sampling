import numpy as np, probe
np.set_printoptions(precision=1,suppress=True,linewidth=160)
rng=np.random.default_rng(12); N=100000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
o=np.argsort(-x,1)
for i in [0,1,2,3,6,7]:
    T=np.zeros((10,10))
    for a in range(10):
        for b in range(10):
            m=(o[:,0]==a)&(o[:,1]==b)
            T[a,b]=h[m,i].mean() if m.any() else np.nan
    print('neuron',i,'mean h by amax(row) x asec(col)'); print(T)
