import numpy as np, probe, sys
from lib import *
rng=np.random.default_rng(4)
x=rng.standard_normal((30000,10))
H=probe.hidden(x); F=lib(x)
i=int(sys.argv[1]); k=int(sys.argv[2])
h=H[:,i]
d1=F['x%d'%k]-F['mo%d'%k]; d2=F['x%d'%k]-F['so%d'%k]
print("bin by x_k - max others: mean h, std")
for lo in np.arange(-3,3,0.5):
    m=(d1>=lo)&(d1<lo+0.5)
    if m.sum()>20: print("%5.1f %6d %.2f %.2f"%(lo,m.sum(),h[m].mean(),h[m].std()))
print("bin by x_k - second others")
for lo in np.arange(-3,3,0.5):
    m=(d2>=lo)&(d2<lo+0.5)
    if m.sum()>20: print("%5.1f %6d %.2f %.2f"%(lo,m.sum(),h[m].mean(),h[m].std()))
print("bin by x_k")
for lo in np.arange(-3,3,0.5):
    m=(x[:,k]>=lo)&(x[:,k]<lo+0.5)
    if m.sum()>20: print("%5.1f %6d %.2f %.2f"%(lo,m.sum(),h[m].mean(),h[m].std()))
