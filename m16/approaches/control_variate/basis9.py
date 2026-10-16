"""Model-free features of the prefix x0..x8 (for control variates on Q = P(correct | x0..x8))."""
import numpy as np
from scipy.special import ndtr
GB=np.array([0.005,0.01,0.02,0.035,0.05,0.075,0.1,0.15,0.2,0.3,0.5])
RB=np.array([0.2,0.4,0.6,0.8])
def onehot(k,K):
    M=np.zeros((len(k),K)); M[np.arange(len(k)),k]=1; return M
def basis9(x, kind="B"):
    x=x[:,:9]; N=len(x)
    o=np.argsort(-x,1); s=np.take_along_axis(x,o,1)
    m1,m2,m3=s[:,0],s[:,1],s[:,2]
    pA=1-ndtr(m1); pB=ndtr(m1)-ndtr(m2); pC=ndtr(m2)      # x9 above both / between / below both
    g12=(m1-m2)/np.maximum(np.abs(m1),1e-12); g23=(m2-m3)/np.maximum(np.abs(m2),1e-12)
    b12=onehot(np.searchsorted(GB,g12),len(GB)+1); b23=onehot(np.searchsorted(GB,g23),len(GB)+1)
    r=np.clip(m2/np.maximum(m1,1e-12),-1,1); br=onehot(np.searchsorted(RB,r),len(RB)+1)
    bl=(m2<=0).astype(float)
    cols=[pA[:,None],pB[:,None],pC[:,None],
          pC[:,None]*b23[:,1:],pC[:,None]*b12[:,1:],pB[:,None]*b12[:,1:],pA[:,None]*b12[:,1:],
          pC[:,None]*br[:,1:],pB[:,None]*br[:,1:],bl[:,None],pC[:,None]*bl[:,None]]
    if kind=="B":
        P1=onehot(o[:,0],9)[:,1:]; P2=onehot(o[:,1],9)[:,1:]
        cols+=[pC[:,None]*P1,pC[:,None]*P2,pB[:,None]*P1,pA[:,None]*P1,pA[:,None]*P2]
    return np.hstack(cols)
