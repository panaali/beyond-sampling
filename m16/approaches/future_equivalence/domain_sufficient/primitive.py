"""Sufficient exact equality of vector ReLU margins on two positive hulls."""
from pathlib import Path
from fractions import Fraction as F
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from canonical import canonical,Ledger,pack


def decode(net,d):
    B=[tuple(map(F,r)) for r in net['B']];V=[tuple(map(F,r)) for r in net['V']]
    if not V or any(len(r)!=d for r in B) or any(len(r)!=len(B) for r in V):raise ValueError('network shape')
    L=[tuple(map(F,r)) for r in net.get('L',[[0]*d for _ in V])]
    if len(L)!=len(V) or any(len(r)!=d for r in L):raise ValueError('linear shape')
    return B,V,L


def restrict_difference(a,b,rays):
    R=[tuple(map(F,r)) for r in rays]
    if not R or not R[0] or any(len(r)!=len(R[0]) for r in R):raise ValueError('nonempty equal-width hull required')
    d=len(R[0]);Ba,Va,La=decode(a,d);Bb,Vb,Lb=decode(b,d)
    if len(Va)!=len(Vb):raise ValueError('output dimension')
    K=len(Va);C=Ledger();B=Ba+Bb
    V=[list(va)+[C.op('mul',F(-1),v) for v in vb] for va,vb in zip(Va,Vb,strict=True)]
    L=[[C.op('sub',x,y) for x,y in zip(la,lb,strict=True)] for la,lb in zip(La,Lb,strict=True)]
    kept=[];columns=[];records=[];comparison_allowance=0
    for i,form in enumerate(B):
        dots=[]
        for ray in R:
            z=F(0)
            for x,y in zip(form,ray,strict=True):z=C.op('add',z,C.op('mul',x,y))
            dots.append(z)
        comparison_allowance+=2*len(dots)
        nonnegative=all(x>=0 for x in dots);nonpositive=all(x<=0 for x in dots)
        if nonpositive:branch='drop'
        elif nonnegative:
            branch='linear'
            for k in range(K):
                for j in range(d):L[k][j]=C.op('add',L[k][j],C.op('mul',V[k][i],form[j]))
        else:
            branch='keep';kept.append(form);columns.append([V[k][i] for k in range(K)])
        records.append(dict(form=form,coefficients=[V[k][i] for k in range(K)],dots=dots,branch=branch))
    coefficients=[[column[k] for column in columns] for k in range(K)]
    normal=canonical(kept,coefficients,d,L)
    linear,hinges=normal['key'];certified=not hinges and not any(x for row in linear for x in row)
    return dict(certified=certified,status='certified_equal' if certified else 'unknown',rays=R,hinges=records,simplified=dict(B=kept,V=coefficients,L=L),canonical=normal,restriction_counts=dict(C.counts),restriction_observed_bits=C.bits,sign_comparison_allowance=comparison_allowance)


def certify(a,b,rays_a,rays_b):
    left=restrict_difference(a,b,rays_a);right=restrict_difference(a,b,rays_b)
    if len(left['rays'][0])!=len(right['rays'][0]):raise ValueError('hull dimension mismatch')
    yes=left['certified'] and right['certified']
    return dict(certified=yes,status='certified_equal' if yes else 'unknown',domains=[left,right])
