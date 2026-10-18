"""Pair adapter consumes independently checked exact sector certificates."""
from fractions import Fraction as F
import mpmath as mp
from primitive import prepare,evaluate,mpq

def unpack(x):
    if isinstance(x,list):
        if len(x)==2 and all(isinstance(s,str) and s.lstrip('-').isdigit() for s in x):return F(int(x[0]),int(x[1]))
        return [unpack(v) for v in x]
    if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
    return x

def prepare_pair(sector,v,w):
    if not any(v) or not any(w):return dict(status='zero_product',counts={},bits=[0,0])
    return prepare(sector['normals']+[v,w])

def evaluate_pair(c,v,w,dps):
    if c['status']=='zero_product':return dict(value='0.0',matrix_result=None,contraction_named_mp={})
    if c['status']=='unsupported_lineality':return dict(status='unsupported_lineality',value=None)
    m=evaluate(c,dps)
    with mp.workdps(dps):
        total=mp.mpf(0)
        for i in range(3):
            for j in range(3):total+=mpq(v[i])*mp.mpf(m['matrix'][i][j])*mpq(w[j])
        return dict(value=mp.nstr(total,dps),matrix_result=m,contraction_named_mp={'mul':18,'add':9})
