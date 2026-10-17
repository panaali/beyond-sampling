"""Exact-zero output-margin ownership only; nonzero clipping inherited unchanged."""
import numpy as np
from families import engine

def cell_error(c,g,A,Wo,truth,ops):
    engine.STATS['cells']+=1
    if g.sum()==0:return 0.0 if truth==0 else engine.mass(c,ops)
    L=Wo@(g[:,None]*A);ops.add(Wo.shape[0]*int(g.sum())*A.shape[1])
    err=0.;q=c
    for o in range(Wo.shape[0]):
        if o==truth:continue
        n=L[truth]-L[o];ops.add(len(n))
        if np.all(n==0):
            if o<truth:return err+engine.mass(q,ops)
            continue
        ci,co=engine.clip(q,n,ops)
        if co is not None:err+=engine.mass(co,ops)
        q=ci
        if q is None:break
    return err

def activate():engine._cell_error=cell_error
