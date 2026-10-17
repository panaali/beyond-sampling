"""Exact dyadic RNN margin derivatives and mask-independent envelopes.
Named arithmetic counts exclude comparisons, indexing, gcd, conversion and control.
"""
from fractions import Fraction as F
from collections import Counter

class Arithmetic:
    def __init__(self):self.counts=Counter();self.bits=0
    def keep(self,x):self.bits=max(self.bits,abs(x).bit_length());return x
    def add(self,a,b):self.counts['add']+=1;return self.keep(a+b)
    def mul(self,a,b):self.counts['mul']+=1;return self.keep(a*b)
    def sub(self,a,b):self.counts['sub']+=1;return self.keep(a-b)
    def dot(self,a,b):
        assert len(a)==len(b)
        out=0
        for x,y in zip(a,b):out=self.add(out,self.mul(x,y))
        return out

def integer_model(model):
    wi,wh,wo=model;wi=list(map(F,wi));wh=[list(map(F,r)) for r in wh];wo=[list(map(F,r)) for r in wo]
    h=len(wi);T=len(wo);assert h and T and all(len(r)==h for r in wh+wo) and len(wh)==h
    vals=wi+sum(wh,[])+sum(wo,[]);D=max(v.denominator for v in vals)
    assert D>0 and D&(D-1)==0 and all(D%v.denominator==0 for v in vals)
    def conv(row):return [int(v*D) for v in row]
    return dict(wi=conv(wi),wh=list(map(conv,wh)),wo=list(map(conv,wo)),D=D,H=h,T=T)

def envelope(m):
    A=Arithmetic();wi,wh,wo,D=m['wi'],m['wh'],m['wo'],m['D'];T=m['T']
    u=list(map(abs,wi));rows=[];den=A.mul(D,D)
    for age in range(T):
        bounds=[]
        for a in range(T):
            for b in range(a):
                d=[abs(A.sub(x,y)) for x,y in zip(wo[a],wo[b])]
                bounds.append(dict(a=a,b=b,bound=F(A.dot(d,u),den)))
        rows.append(dict(age=age,pairs=bounds,maximum=max((r['bound'] for r in bounds),default=F(0))))
        if age<T-1:
            u=[A.dot(list(map(abs,row)),u) for row in wh];den=A.mul(den,D)
    return dict(rows=rows,counts=dict(A.counts),bits=A.bits)

def witness(m,x,upper):
    A=Arithmetic();wi,wh,wo,D=m['wi'],m['wh'],m['wo'],m['D'];H,T=m['H'],m['T']
    assert len(x)==T and all(type(v) is int for v in x)
    h=[0]*H;J=[[0]*T for _ in range(H)];scale=1;history=[];boundaries=[];zeros=[];radius=[]
    for t in range(T):
        B=[]
        for i in range(H):
            row=[]
            for j in range(T):
                v=A.dot(wh[i],[J[k][j] for k in range(H)])
                if j==t:v=A.add(v,A.mul(wi[i],scale))
                row.append(v)
            B.append(row)
        v=[A.add(A.dot(wh[i],h),A.mul(A.mul(wi[i],x[t]),scale)) for i in range(H)]
        assert v==[A.dot(row,x) for row in B]
        gates=[z>0 for z in v];history.append(gates)
        for i,(row,z) in enumerate(zip(B,v)):
            if not any(row):zeros.append([t,i]);assert z==0
            elif z==0:boundaries.append([t,i])
            else:
                norm=0
                for b in row:norm=A.add(norm,abs(b))
                radius.append(F(abs(z),A.mul(2,norm)))
        J=[r if gates[i] else [0]*T for i,r in enumerate(B)];h=[max(z,0) for z in v];scale=A.mul(scale,D)
    L=[[A.dot(row,[J[k][j] for k in range(H)]) for j in range(T)] for row in wo]
    logits=[A.dot(row,h) for row in wo];assert logits==[A.dot(row,x) for row in L]
    den=A.mul(scale,D);sens=[]
    for j in range(T):
        col=[r[j] for r in L];s=F(A.sub(max(col),min(col)),den)
        age=T-1-j;assert s<=upper['rows'][age]['maximum']
        sens.append(s)
    gaps=[abs(A.sub(x[i],x[j])) for i in range(T) for j in range(i)]
    rank_strict=all(gaps)
    if rank_strict:radius.append(F(min(gaps),4))
    strict=not boundaries and rank_strict
    return dict(x=x,history=history,identically_zero_gates=zeros,boundary_gates=boundaries,strict=strict,
        radius=min(radius) if strict and radius else None,logit_map_numerators=L,logit_denominator=den,
        sensitivities_by_input=sens,age_sensitivities=list(reversed(sens)),
        local_derivative_certified=strict,counts=dict(A.counts),bits=A.bits)

def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x
