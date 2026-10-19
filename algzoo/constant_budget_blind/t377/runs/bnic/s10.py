import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
M=np.maximum.accumulate(x,1); s=np.argsort(-x,1); am=s[:,0]
age=9-am
for i in [3,6,12,8,14,5]:
    print('neuron',i)
    for a in range(10):
        k=(age==a)
        if i==8: f=M[:,9]-x[:,7]
        elif i==14: f=M[:,9]-x[:,4]
        elif i==5: f=np.maximum(x[:,9]-M[:,8],0)
        else: f=M[:,8]-x[:,9]
        kk=k&(f>0.05)
        c=(h[kk,i]*f[kk]).sum()/(f[kk]**2).sum(); r=1-((h[kk,i]-c*f[kk])**2).mean()/h[kk,i].var() if kk.sum()>10 else 0
        print(f' age{a} n={k.sum()} mean={h[k,i].mean():.1f} slope={c:.1f} r2={r:.2f}')
