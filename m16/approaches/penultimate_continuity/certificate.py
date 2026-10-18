"""Sufficient exact slope criterion for conditional hidden-state continuity."""
from fractions import Fraction as F
from itertools import product


def rational_model(model):
    wi,wh,wo=model
    return tuple(F(float(x)) for x in wi),tuple(tuple(F(float(x)) for x in r) for r in wh),tuple(tuple(F(float(x)) for x in r) for r in wo)


def certify(model,label):
    wi,wh,wo=model
    if len(wi)!=2 or len(wh)!=2 or any(len(r)!=2 for r in wh+wo):raise ValueError('H2 required')
    if not 0<=label<len(wo):raise ValueError('invalid target label')
    M=max(sum(map(abs,r)) for r in wh)
    if M==0:return dict(sufficient=True,state_independent=True,M=M,G=F(0),gates=[],margins=[],unknown=[])
    G=F(0);gates=[];margins=[];unknown=[]
    for i,slope in enumerate(wi):
        if slope==0:
            unknown.append(dict(kind='gate',neuron=i));continue
        coefficient=2*M**3/abs(slope);G+=coefficient
        gates.append(dict(neuron=i,slope=slope,coefficient=coefficient))
    for support in product((False,True),repeat=2):
        if not any(support):continue
        for other in range(len(wo)):
            if other==label:continue
            delta=tuple(a-b for a,b in zip(wo[label],wo[other]))
            if all(not support[i] or delta[i]==0 for i in range(2)):
                margins.append(dict(support=support,competitor=other,permanent_tie=True,
                                    target_wins_tie=label<other,coefficient=F(0)))
                continue
            slope=sum(delta[i]*wi[i] for i in range(2) if support[i])
            if slope==0:
                unknown.append(dict(kind='margin',support=support,competitor=other,restricted_delta=tuple(delta[i] if support[i] else F(0) for i in range(2))))
                continue
            coefficient=2*sum(map(abs,delta))*M**3/abs(slope);G+=coefficient
            margins.append(dict(support=support,competitor=other,permanent_tie=False,
                                slope=slope,coefficient=coefficient))
    return dict(sufficient=not unknown,state_independent=False,M=M,G=G if not unknown else None,
                partial_G=G,gates=gates,margins=margins,unknown=unknown)


def serial(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [serial(v) for v in x]
    return x
