import numpy as np, probe, sys
from fx import ns
x=np.random.default_rng(61).standard_normal((100000,10)); H=probe.hidden(x); d=ns(x)
i=int(sys.argv[1]); h=H[:,i]
print("mean h by am9 (rows) x a29 (cols)")
for a in range(9):
    print(a," ".join("%5.2f"%h[(d['am9']==a)&(d['a29']==b)].mean() if a!=b else "  -  " for b in range(9)))
