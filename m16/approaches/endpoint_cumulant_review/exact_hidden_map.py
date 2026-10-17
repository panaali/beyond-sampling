"""Exact dyadic fixed-map/rank checks at a supplied interior point; no cone enumeration."""
from fractions import Fraction as F
import itertools
import numpy as np

def matrix(model,point):
    Wi,Wh,Wo=[np.asarray(x,dtype=object) for x in model]
    Wi=np.vectorize(lambda x:F(float(x)))(Wi);Wh=np.vectorize(lambda x:F(float(x)))(Wh)
    x=np.array([F(float(v)) for v in point],dtype=object)
    A=np.full((len(Wi),len(x)),F(0),dtype=object);masks=[]
    for t in range(len(x)):
        A=Wh@A;A[:,t]+=Wi
        g=A@x>0;masks.append(g.tolist());A[~g,:]=F(0)
    return A,masks

def rank(A):
    if not any(A.ravel()):return 0
    return 2 if any(A[0,i]*A[1,j]!=A[1,i]*A[0,j] for i,j in itertools.combinations(range(A.shape[1]),2)) else 1

def ray_winner(model,A,point):
    x=np.array([F(float(v)) for v in point],dtype=object);h=A@x
    assert min(h)>=0 and max(h)>0
    Wo=np.vectorize(lambda x:F(float(x)))(np.asarray(model[2],dtype=object));logits=Wo@h
    return max(range(len(logits)),key=lambda j:logits[j])
