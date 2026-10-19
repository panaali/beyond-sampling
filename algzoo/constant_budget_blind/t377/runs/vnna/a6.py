import numpy as np, probe
np.set_printoptions(precision=1,suppress=True,linewidth=220)
base=-0.5
for A in [1.0,2.0]:
    x=np.full((10,10),base)
    for k in range(10): x[k,k]=A
    H=probe.hidden(x)
    print("spike A=",A,"base",base,"rows=neurons, cols=spike position")
    for i in range(32): print("%2d"%i,H[:,i])
