"""Exact conservative Gaussian strips over extended recurrent gate words."""
from fractions import Fraction as F
from itertools import product

def mm(A,B):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*B)] for row in A]
def mv(A,b):return [sum(x*y for x,y in zip(row,b)) for row in A]
def masked(A,g):return [[x*int(on) for x in row] for row,on in zip(A,g)]
def key(row):
    pivot=next((v for v in row if v),None)
    return tuple(v/pivot for v in row) if pivot else tuple(row)

def count_forms(forms):
    literal=F(0);unique={};unknown=[];zero_hidden=0;counted=[]
    for idx,f in enumerate(forms):
        A,B,C0,C1,E=f['row'];state=abs(C0)+abs(C1);normal=max(abs(A),abs(B))
        if not state:zero_hidden+=1;continue
        if not normal:unknown.append(dict(index=idx,**f));continue
        value=state/normal;literal+=value;k=key(f['row']);unique[k]=value;counted.append(dict(index=idx,coefficient=value))
    return dict(sufficient=not unknown,K_literal=None if unknown else literal,K_unique=None if unknown else sum(unique.values(),F(0)),partial_literal=literal,partial_unique=sum(unique.values(),F(0)),emitted=len(forms),nonzero_hidden=len(forms)-zero_hidden,zero_hidden=zero_hidden,unique_known=len(unique),unknown=unknown,counted=counted)

def certify(model,label):
    wi,wh,wo=model
    if len(wi)!=2 or len(wh)!=2 or any(len(row)!=2 for row in wh+wo):raise ValueError('H2 required')
    if not 0<=label<len(wo):raise ValueError('target label out of bounds')
    if all(row==wo[0] for row in wo):return dict(sufficient=True,K_literal=F(0),K_unique=F(0),constant_output=True,forms=[],emitted=0,nonzero_hidden=0,zero_hidden=0,unique_known=0,unknown=[])
    forms=[]
    for first,middle in product(product((False,True),repeat=2),repeat=2):
        fC=masked(wh,first);fA=[v*int(g) for v,g in zip(wi,first)]
        mC=masked(mm(wh,fC),middle);mA=[v*int(g) for v,g in zip(mv(wh,fA),middle)];mE=[v*int(g) for v,g in zip(wi,middle)]
        C=mm(wh,mC);A=mv(wh,mA);E=mv(wh,mE)
        Z=[(A[i],wi[i],C[i][0],C[i][1],E[i]) for i in range(2)]
        for i,row in enumerate(Z):forms.append(dict(first=first,middle=middle,kind='final_gate',neuron=i,row=row))
        for final in product((False,True),repeat=2):
            if not any(final):continue
            for other in range(len(wo)):
                if other==label:continue
                delta=[(a-b)*int(g) for a,b,g in zip(wo[label],wo[other],final)]
                if not any(delta):continue # Permanent support-identical tie, ownership unchanged.
                row=tuple(sum(delta[i]*Z[i][j] for i in range(2)) for j in range(5))
                forms.append(dict(first=first,middle=middle,final=final,kind='margin',competitor=other,row=row))
    return dict(**count_forms(forms),forms=forms)

def serial(v):
    if isinstance(v,F):return str(v)
    if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    return v
