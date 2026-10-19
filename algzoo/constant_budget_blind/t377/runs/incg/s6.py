import numpy as np, probe
# crafted: zero-prefix = shorter sequence. single impulse at position j with value v, rest zero
for j in [0,4,8,9]:
    for v in [1.0,-1.0]:
        x=np.zeros((1,10),np.float32); x[0,j]=v
        print(j,v,np.round(np.asarray(probe.hidden(x))[0],2))
