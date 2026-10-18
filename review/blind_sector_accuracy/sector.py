"""Exact orthant identity and binary-correctness interval calculations."""
from fractions import Fraction as F
from itertools import permutations

def orthant(rays):
    if type(rays)is not list or len(rays)!=4 or any(type(r)is not list or len(r)!=4 for r in rays):raise ValueError('four by four rays')
    if any(type(x)is not int for r in rays for x in r):raise ValueError('integer rays')
    key=0
    for j in range(4):
        col=[r[j] for r in rays]
        if min(col)<0<max(col):raise ValueError('mixed coordinate signs')
        if not any(col):raise ValueError('zero coordinate span')
        if max(col)>0:key|=1<<j
    det=0
    for perm in permutations(range(4)):
        term=1
        for i,j in enumerate(perm):term*=rays[i][j]
        det+=(-term if sum(perm[i]>perm[j] for i in range(4) for j in range(i+1,4))%2 else term)
    if not det:raise ValueError('singular simplex')
    denom=1
    for r in rays:denom*=sum(abs(x) for x in r)
    return key,F(abs(det),denom)

def binary_law(lo,hi,p=F(1,16)):
    if any(type(x)is not F for x in [lo,hi,p]) or not 0<=lo<=hi<=p or p<=0:raise ValueError('physical probability interval')
    f=lambda x:min(x,p-x)
    losslo=min(f(lo),f(hi));losshi=p/2 if lo<=p/2<=hi else max(f(lo),f(hi))
    return {'correct':[lo,hi],'error':[p-hi,p-lo],'conditional_correct':[lo/p,hi/p],'best_binary_loss':[losslo,losshi],'strictly_nonconstant':lo>0 and hi<p}
