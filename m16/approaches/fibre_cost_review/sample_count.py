"""Independent scalar sampler inventory; labels/RNG free, first h0 multiply skipped."""
from collections import Counter
class Scalar:
    counts=Counter()
    def __init__(self,value):self.value=float(value)
    def __add__(self,other):self.counts['add']+=1;return Scalar(self.value+other.value)
    def __mul__(self,other):self.counts['mul']+=1;return Scalar(self.value*other.value)
    def __gt__(self,other):self.counts['comparison']+=1;return self.value>other.value

def count(model,x=(.3,-1.2,.7),truth=0):
    Scalar.counts=Counter();wi,wh,wo=model;H=len(wi);T=len(wo);wi=[Scalar(z) for z in wi];wh=[[Scalar(z) for z in row] for row in wh];wo=[[Scalar(z) for z in row] for row in wo];x=list(map(Scalar,x));zero=Scalar(0)
    h=[x[0]*z for z in wi];h=[z if z>zero else zero for z in h]
    for t in range(1,T):
        new=[]
        for i in range(H):
            value=x[t]*wi[i]
            for j in range(H):value=value+wh[i][j]*h[j]
            new.append(value if value>zero else zero)
        h=new
    logits=[]
    for row in wo:
        value=row[0]*h[0]
        for j in range(1,H):value=value+row[j]*h[j]
        logits.append(value)
    winner=0
    for i in range(1,T):
        if logits[i]>logits[winner]:winner=i
    # Integer correctness comparison counted explicitly; target computation is free.
    Scalar.counts['comparison']+=1;correct=winner==truth
    result=dict(Scalar.counts);result['total']=sum(Scalar.counts.values());result['prediction']=winner;result['correct']=correct
    expected=4*T*H+2*(T-1)*H*H
    assert result['total']==expected
    result['formula']=expected;result['label_generation_free']=True;return result
