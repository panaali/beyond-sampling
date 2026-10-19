import numpy as np
def pf(x):
    P=x[:,:9]; s=-np.sort(-P,1); o=np.argsort(-P,1)
    return dict(M8=s[:,0],s8=s[:,1],t8=s[:,2],p8=o[:,0],q8=o[:,1],x9=x[:,9],x8=x[:,8],
                M7=x[:,:8].max(1),M=x.max(1),mean8=P.mean(1))
def relufit(A,y,it=8):
    g=y>0.3
    for _ in range(it):
        c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None)
        pr=A@c; g2=(y>0.3)|(pr>0)
        if (g2==g).all(): break
        g=g2
    pr=np.maximum(A@c,0)
    return c,1-((pr-y)**2).mean()/y.var()
