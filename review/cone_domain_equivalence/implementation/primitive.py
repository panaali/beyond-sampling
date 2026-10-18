"""Exact span restriction; finite positive hulls only, no trained application."""
from fractions import Fraction as F
from frozen_canonical import canonical,Ledger,pack


def matrix(rows):return [list(map(F,row)) for row in rows]
def decode(net,n):
    B=matrix(net['B']);V=matrix(net['V'])
    if not V or any(len(r)!=n for r in B) or any(len(r)!=len(B) for r in V):raise ValueError('network shape')
    L=matrix(net.get('L',[[0]*n for _ in V]))
    if len(L)!=len(V) or any(len(r)!=n for r in L):raise ValueError('linear shape')
    return dict(B=B,V=V,L=L)

def rref(rows,limit,C):
    A=[list(r) for r in rows];h=len(A);w=len(A[0]) if h else 0;pivots=[];rank=0
    assert all(len(r)==w for r in A) and 0<=limit<=w
    for col in range(limit):
        pivot=next((i for i in range(rank,h) if A[i][col]),None)
        if pivot is None:continue
        A[rank],A[pivot]=A[pivot],A[rank];v=A[rank][col]
        A[rank]=[C.op('div',x,v) for x in A[rank]]
        for i in range(h):
            if i==rank or not A[i][col]:continue
            factor=A[i][col];A[i]=[C.op('sub',x,C.op('mul',factor,y)) for x,y in zip(A[i],A[rank],strict=True)]
        pivots.append(col);rank+=1
        if rank==h:break
    return A,pivots

def span(R,C):
    n=len(R[0]);m=len(R)
    _,columns=rref([[R[i][j] for i in range(m)] for j in range(n)],m,C)
    basis=[R[i] for i in columns];d=len(basis)
    if not d:
        assert all(not any(r) for r in R)
        return dict(basis=[],coordinates=[[] for _ in R],indices=[],dimension=0)
    augmented=[[basis[i][j] for i in range(d)]+[R[i][j] for i in range(m)] for j in range(n)]
    reduced,pivots=rref(augmented,d,C);assert pivots==list(range(d))
    assert all(not any(row) for row in reduced[d:])
    coords=[[reduced[j][d+i] for j in range(d)] for i in range(m)]
    reconstructed=[[sum((z*b[j] for z,b in zip(c,basis,strict=True)),F(0)) for j in range(n)] for c in coords]
    assert reconstructed==R
    return dict(basis=basis,coordinates=coords,indices=columns,dimension=d)

def project(net,basis,C):
    def rows(a):
        out=[]
        for row in a:
            r=[]
            for b in basis:
                v=F(0)
                for x,y in zip(row,b,strict=True):v=C.op('add',v,C.op('mul',x,y))
                r.append(v)
            out.append(r)
        return out
    return dict(B=rows(net['B']),V=net['V'],L=rows(net['L']))

def on_domain(a,b,rays):
    R=matrix(rays)
    if not R or not R[0] or any(len(r)!=len(R[0]) for r in R):raise ValueError('nonempty equal-width hull required')
    n=len(R[0]);a=decode(a,n);b=decode(b,n)
    if len(a['V'])!=len(b['V']):raise ValueError('output dimensions')
    C=Ledger();S=span(R,C);base=dict(rays=R,original_a=a,original_b=b,span=S)
    if S['dimension']==0:
        return dict(base,certified=True,status='equal',projected_a=None,projected_b=None,canonical=None,signs=[],arithmetic=dict(C.counts),observed_bits=C.bits,sign_comparison_allowance=0)
    pa=project(a,S['basis'],C);pb=project(b,S['basis'],C);d=S['dimension'];K=len(pa['V'])
    forms=pa['B']+pb['B'];V=[va+[C.op('mul',F(-1),x) for x in vb] for va,vb in zip(pa['V'],pb['V'],strict=True)]
    L=[[C.op('sub',x,y) for x,y in zip(la,lb,strict=True)] for la,lb in zip(pa['L'],pb['L'],strict=True)]
    kept=[];cols=[];signs=[]
    for i,form in enumerate(forms):
        dots=[]
        for ray in S['coordinates']:
            v=F(0)
            for x,y in zip(form,ray,strict=True):v=C.op('add',v,C.op('mul',x,y))
            dots.append(v)
        nonpos=all(x<=0 for x in dots);nonneg=all(x>=0 for x in dots)
        if nonpos:branch='drop'
        elif nonneg:
            branch='linear'
            for k in range(K):
                for j in range(d):L[k][j]=C.op('add',L[k][j],C.op('mul',V[k][i],form[j]))
        else:branch='keep';kept.append(form);cols.append([V[k][i] for k in range(K)])
        signs.append(dict(form=form,coefficients=[V[k][i] for k in range(K)],dots=dots,branch=branch))
    result=canonical(kept,[[col[k] for col in cols] for k in range(K)],d,L)
    linear,hinges=result['key'];equal=not hinges and not any(x for row in linear for x in row)
    return dict(base,certified=equal,status='equal' if equal else 'unequal_on_represented_hull',projected_a=pa,projected_b=pb,canonical=result,signs=signs,arithmetic=dict(C.counts),observed_bits=C.bits,sign_comparison_allowance=2*len(forms)*len(R))

def certify(a,b,rays_a,rays_b):
    left=on_domain(a,b,rays_a);right=on_domain(a,b,rays_b)
    if len(left['rays'][0])!=len(right['rays'][0]):raise ValueError('hull dimension mismatch')
    equal=left['certified'] and right['certified']
    return dict(certified=equal,status='equal_on_union' if equal else 'unequal_on_represented_union',domains=[left,right])
