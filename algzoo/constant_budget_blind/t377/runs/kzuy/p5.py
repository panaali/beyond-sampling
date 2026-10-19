import numpy as np, probe
from dic2 import dic2
from dic import greedy
x=np.random.default_rng(3).standard_normal((20000,10))
h=probe.hidden(x); D=dic2(x)
for i in range(16):
    print(i,' '.join(f'{k}({v:.3f})' for k,v in greedy(D,h[:,i],5)),flush=True)
