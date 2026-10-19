import numpy as np, probe
from fx import ns
rng=np.random.default_rng(31); x=rng.standard_normal((40000,10)); H=probe.hidden(x); d=ns(x)
h=H[:,14]; P1=d['P'](8,1); P2=d['P'](8,2); x8=x[:,8]
u=x8-P2; v=x8-P1
print("rows: x8-P2 bins; cols: x8-P1 bins (mean h)")
vb=np.arange(-3,1.6,0.5)
print("      "+" ".join("%5.1f"%b for b in vb))
for lo in np.arange(-2.5,2.6,0.5):
    row=[]
    for vl in vb:
        m=(u>=lo)&(u<lo+0.5)&(v>=vl)&(v<vl+0.5)
        row.append("%5.2f"%h[m].mean() if m.sum()>30 else "   - ")
    print("%5.1f "%lo+" ".join(row))
