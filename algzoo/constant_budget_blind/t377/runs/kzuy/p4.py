import numpy as np, probe
from dic import dic,greedy
x=np.random.default_rng(3).standard_normal((20000,10))
h=probe.hidden(x); D=dic(x)
for i in range(16):
    print(i,' '.join(f'{k}({v:.3f})' for k,v in greedy(D,h[:,i])),flush=True)
