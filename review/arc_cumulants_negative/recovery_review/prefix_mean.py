"""Exact-real-arithmetic h2 means from two Gaussian inputs; scalar source cost inventory."""
from collections import Counter
import math
import numpy as np
CONST=1/(2*math.sqrt(2*math.pi))
def term(a,b,ops):
    ops['hypot']+=1;r=math.hypot(a,b)
    ops['comparison']+=1
    if a<0:
        ops['arithmetic']+=3
        return b*b/(r-a)
    ops['arithmetic']+=1
    return r+a

def means(wi,wh):
    wi=np.asarray(wi,dtype=float);wh=np.asarray(wh,dtype=float);H=len(wi);assert wh.shape==(H,H)
    ops=Counter(comparison=2*H,negation=H,arithmetic=2*(H*H+H*(H-1)))
    pos=np.maximum(wi,0);neg=np.maximum(-wi,0);ap=wh@pos;an=wh@neg
    out=[]
    for a,c,b in zip(ap,an,wi):
        x=term(float(a),float(b),ops);y=term(float(c),float(b),ops);out.append((x+y)*CONST);ops['arithmetic']+=2
    return np.asarray(out),ap,an,dict(ops)
