"""Independent exhaustive small rational vertex reference; no candidate imports."""
from fractions import Fraction as F
from itertools import combinations

def solve(A,b):
    n=len(b);rows=[[F(x) for x in r]+[F(y)] for r,y in zip(A,b,strict=True)]
    for col in range(n):
        pivot=next((r for r in range(col,n) if rows[r][col]),None)
        if pivot is None:return None
        rows[col],rows[pivot]=rows[pivot],rows[col]
        z=rows[col][col];rows[col]=[x/z for x in rows[col]]
        for r in range(n):
            if r!=col:
                z=rows[r][col];rows[r]=[x-z*y for x,y in zip(rows[r],rows[col],strict=True)]
    return tuple(rows[r][-1] for r in range(n))

def vertices(box,coeff,constant,leader,second,gap):
    n=len(box);constraints=[];g=F(gap)
    for j,(l,u) in enumerate(box):
        a=[F(0)]*n;a[j]=1;constraints.append((a,F(u)))
        a=[F(0)]*n;a[j]=-1;constraints.append((a,-F(l)))
    a=[F(0)]*n;a[second]=1;a[leader]=-1;constraints.append((a,-g))
    for r in range(n):
        if r in (leader,second):continue
        a=[F(0)]*n;a[r]=1;a[second]=-1;constraints.append((a,-g))
    vs=set();systems=0;nonsingular=0
    for inds in combinations(range(len(constraints)),n):
        systems+=1;x=solve([constraints[i][0] for i in inds],[constraints[i][1] for i in inds])
        if x is None:continue
        nonsingular+=1
        if all(sum((z*w for z,w in zip(a,x,strict=True)),F(0))<=b for a,b in constraints):vs.add(x)
    ordered=sorted(vs);a=list(map(F,coeff));c=F(constant);values=[c+sum((u*v for u,v in zip(a,x,strict=True)),F(0)) for x in ordered]
    return dict(vertices=ordered,values=values,range=None if not values else [min(values),max(values)],systems=systems,nonsingular=nonsingular,constraints=constraints)
