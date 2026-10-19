import numpy as np, probe
def traj(x):
    N=len(x); H=np.zeros((N,11,8))
    for t in range(1,11):
        z=np.zeros_like(x); z[:,10-t:]=x[:,:t]; H[:,t]=probe.hidden(z)
    return H
