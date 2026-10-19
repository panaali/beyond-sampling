import numpy as np, probe
np.set_printoptions(precision=2,suppress=True,linewidth=150)
for a in [1,-1]:
    x=np.zeros((10,10))
    for t in range(10): x[t,t]=a
    print('impulse',a); print(probe.hidden(x))
