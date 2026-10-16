"""Model-free feature bases for regression control variates (expectations known model-free)."""
import numpy as np
T=10
GB=np.array([0.005,0.01,0.02,0.035,0.05,0.075,0.1,0.15,0.2,0.3,0.5])
RB=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
def onehot(k,K): 
    M=np.zeros((len(k),K)); M[np.arange(len(k)),k]=1; return M
def basis(x, kind="B"):
    o=np.argsort(-x,1); s=np.take_along_axis(x,o,1)
    v1,v2,v3=s[:,0],s[:,1],s[:,2]
    blind=(v2<=0).astype(float); nb=1-blind
    g12=(v1-v2)/np.abs(v1); g23=(v2-v3)/np.maximum(np.abs(v2),1e-12)
    r21=np.clip(v2/np.maximum(v1,1e-12),0,1)
    b12=onehot(np.searchsorted(GB,g12),len(GB)+1)*nb[:,None]
    b23=onehot(np.searchsorted(GB,g23),len(GB)+1)*nb[:,None]
    br=onehot(np.searchsorted(RB,r21),len(RB)+1)*nb[:,None]
    cols=[blind[:,None],b12[:,1:],b23[:,1:],br[:,1:]]
    if kind in("B","C"):
        P1=onehot(o[:,0],T)[:,1:];P2=onehot(o[:,1],T)[:,1:];P3=onehot(o[:,2],T)[:,1:]
        cols+=[P1*nb[:,None],P2*nb[:,None],P3*nb[:,None]]
    if kind=="C":
        # p2 x small-gap, p1 x small r21 interactions
        sg=(g23<0.1)*nb; sr=(r21<0.5)*nb; s12=(g12<0.1)*nb
        P1=onehot(o[:,0],T);P2=onehot(o[:,1],T)
        cols+=[P2*sg[:,None],P1*sr[:,None],P2*sr[:,None],P1*s12[:,None]]
    return np.hstack(cols)
