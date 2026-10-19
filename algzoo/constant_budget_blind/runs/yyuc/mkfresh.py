import numpy as np, probe
x=np.random.default_rng(7).standard_normal((50000,10)); np.save('x7.npy',x); np.save('H7.npy',probe.hidden(x))
