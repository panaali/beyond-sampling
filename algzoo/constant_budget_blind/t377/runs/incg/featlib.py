import numpy as np
def F(x,n):
    N=len(x)
    if n.startswith("inTop2_"):
        k=int(n[7:]); o=np.sort(np.delete(x,k,1),1); return np.maximum(x[:,k]-o[:,-2],0)
    if n.startswith("isMax"):
        k=int(n[5:]); o=np.delete(x,k,1).max(1); return np.maximum(x[:,k]-o,0)
    if n.startswith("below"):
        k=int(n[5:]); o=np.delete(x,k,1).max(1); return np.maximum(o-x[:,k],0)
    if n.startswith("rec"):
        k=int(n[3:]); return np.maximum(x[:,k]-x[:,:k].max(1),0)
    if n.startswith("mex"):
        k=int(n[3:]); return np.delete(x,k,1).max(1)
    if n.startswith("s2ex"):
        k=int(n[4:]); return np.sort(np.delete(x,k,1),1)[:,-2]
    if n.startswith("gmax"): return x[:,[int(c) for c in n[4:]]].max(1)
    if n.startswith("p2nd"):
        t=int(n[4:]); return np.sort(x[:,:t+1],1)[:,-2]
    if n.startswith("pm"):
        t=int(n[2:]); return x[:,:t+1].max(1)
    if n.startswith("sm"):
        t=int(n[2:]); return x[:,t:].max(1)
    if n.startswith("S"):
        r=int(n[1:]); return -np.sort(-x,1)[:,r]
    if n.startswith("x"): return x[:,int(n[1:])]
    raise KeyError(n)
