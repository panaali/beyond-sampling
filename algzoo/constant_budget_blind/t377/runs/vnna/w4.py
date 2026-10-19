import numpy as np, probe
from fx import ns
from gate import fastfit
np.set_printoptions(precision=2,suppress=True,linewidth=200)
x=np.random.default_rng(62).standard_normal((60000,10)); H=probe.hidden(x); d=ns(x)
h=H[:,4]
m=np.isin(d['a29'],[0,2,4,6])
X=np.c_[x, d['M9'],d['S9'],d['P'](9,3)][m]
c,r=fastfit(h[m],X); print("within even-a29: R2 %.3f"%r, c)
# relative to the position of S9: values right after?
for a in [0,2,4,6]:
    mm=d['a29']==a
    c,r=fastfit(h[mm],np.c_[x,d['S9']][mm]); print(a,"R2 %.3f"%r,c)
