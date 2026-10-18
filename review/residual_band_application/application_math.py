"""Separately timed versions of the reviewed Gaussian band formulas."""
import math
from gaussian import mass

def evaluate(bands,mode):
    if mode not in ('first','second','direct'):raise ValueError('mode')
    values=[]
    for a,b,sign in bands:
        if sign not in (-1,1) or not a<b or a<0<b:raise ValueError('one-sided signed band')
        if mode=='direct':value=mass(a,b)
        else:
            c=a if a>=0 else b;d=b-a if a>=0 else a-b;q=1 if d>0 else -1
            c=float(c);d=float(d);psi=math.exp(-c*c)/math.sqrt(math.pi)
            value=q*psi*d if mode=='first' else q*psi*(d-c*d*d)
        values.append(sign*value)
    return math.fsum(values),values
