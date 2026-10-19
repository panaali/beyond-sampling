import numpy as np, probe
from util import *
rng=np.random.default_rng(5)
x=rng.standard_normal((50000,10)); H=probe.hidden(x)
m8=x[:,:9].max(1); o=np.argsort(-x[:,:9],1); s9=np.take_along_axis(x[:,:9],o,1)
h=H[:,15]; f=np.maximum(x[:,9]-m8,0)
print("15:",np.polyfit(f,h,1), r2(np.polyfit(f,h,1)[0]*f,h))
h=H[:,10]
for k in range(4):
    sub=x[:,9]-s9[:,k]
    for c in [0]:
        f=np.maximum(sub,0); a=(f@h)/(f@f); print("10 relu(x9-%d-th of x0..8)"%(k+1),round(a,2),round(r2(a*f,h),3))
# x9 gated by rank
r9=(x[:,:9]>x[:,9:10]).sum(1)
for k in range(5): print(k, h[r9==k].mean(), (h[r9==k]/np.maximum(x[r9==k,9],1e-9)).mean(), (h[r9==k]>0).mean())
