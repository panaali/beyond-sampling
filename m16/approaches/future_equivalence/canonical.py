"""Exact rational canonical homogeneous ReLU margin functions."""
from fractions import Fraction as F
from collections import Counter
class Ledger:
    def __init__(self):self.counts=Counter();self.bits=[0,0]
    def op(self,k,a,b):
        self.counts[k]+=1
        if k=='add':z=a+b
        elif k=='mul':z=a*b
        elif k=='div':z=a/b
        elif k=='sub':z=a-b
        else:raise ValueError(k)
        self.bits=[max(self.bits[0],abs(z.numerator).bit_length()),max(self.bits[1],z.denominator.bit_length())];return z

def canonical(forms,coefficients,dimension,linear=None):
    H=len(forms);K=len(coefficients);assert dimension>=1 and K>=1
    assert all(len(r)==dimension for r in forms) and all(len(r)==H for r in coefficients)
    B=[tuple(map(F,r)) for r in forms];V=[tuple(map(F,r)) for r in coefficients]
    if linear is None:linear=[[F(0)]*dimension for _ in range(K)]
    assert len(linear)==K and all(len(r)==dimension for r in linear)
    L=[list(map(F,r)) for r in linear];A=Ledger();groups={}
    for i,b in enumerate(B):
        first=next((x for x in b if x),None)
        if first is None:continue
        u=tuple(A.op('div',x,first) for x in b)
        if u not in groups:groups[u]=[F(0)]*K
        for k in range(K):
            c=A.op('mul',abs(first),V[k][i]);groups[u][k]=A.op('add',groups[u][k],c)
            if first<0:
                for j in range(dimension):L[k][j]=A.op('add',L[k][j],A.op('mul',V[k][i],b[j]))
    hinges=tuple((u,tuple(c)) for u,c in sorted(groups.items()) if any(c))
    key=(tuple(tuple(r) for r in L),hinges)
    scale=next((abs(x) for r in L for x in r if x),None)
    if scale is None:scale=next((abs(x) for _,c in hinges for x in c if x),None)
    if scale is None:projective=key
    else:projective=(tuple(tuple(A.op('div',x,scale) for x in r) for r in L),tuple((u,tuple(A.op('div',x,scale) for x in c)) for u,c in hinges))
    return dict(key=key,projective=projective,counts=dict(A.counts),bits=A.bits)

def margins(readout):
    W=[list(map(F,r)) for r in readout];assert len(W)>=2 and all(len(r)==len(W[0]) for r in W)
    return [[x-y for x,y in zip(r,W[0],strict=True)] for r in W[1:]]

def future_forms(Wi,Wh,M):
    Wi=list(map(F,Wi));Wh=[list(map(F,r)) for r in Wh];M=[list(map(F,r)) for r in M];H=len(Wi)
    assert H>=1 and len(Wh)==len(M)==H and all(len(r)==H for r in Wh)
    d=len(M[0]);assert d>=1 and all(len(r)==d for r in M)
    return [[sum((Wh[i][k]*M[k][j] for k in range(H)),F(0)) for j in range(d)]+[Wi[i]] for i in range(H)]

def direct(forms,coefficients,x,linear=None):
    x=list(map(F,x));K=len(coefficients)
    if linear is None:linear=[[F(0)]*len(x) for _ in range(K)]
    vals=[max(F(0),sum((F(a)*b for a,b in zip(r,x,strict=True)),F(0))) for r in forms]
    return [sum((F(a)*b for a,b in zip(r,vals,strict=True)),F(0))+sum((F(a)*b for a,b in zip(l,x,strict=True)),F(0)) for r,l in zip(coefficients,linear,strict=True)]

def from_key(key,x):
    L,hinges=key;x=list(map(F,x));out=[sum((a*b for a,b in zip(r,x,strict=True)),F(0)) for r in L]
    for u,c in hinges:
        v=max(F(0),sum((a*b for a,b in zip(u,x,strict=True)),F(0)))
        out=[a+b*v for a,b in zip(out,c,strict=True)]
    return out

def label(m):
    scores=[F(0)]+list(m);return max(range(len(scores)),key=lambda i:scores[i])

def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x
