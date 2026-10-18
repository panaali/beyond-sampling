"""Exact original-ray full-rank witnesses; no functional or geometry calls."""
from fractions import Fraction as F
from itertools import permutations
from collections import Counter

class Counts:
    def __init__(self): self.n=Counter();self.bits=[0,0]
    def op(self,k,a,b):
        self.n[k]+=1
        z={'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[k]()
        self.bits=[max(self.bits[0],abs(z.numerator).bit_length()),max(self.bits[1],z.denominator.bit_length())]
        return z

def determinant4(rows):
    assert len(rows)==4 and all(len(r)==4 for r in rows)
    C=Counts();total=F(0);terms=[]
    for p in permutations(range(4)):
        product=rows[0][p[0]]
        for i in range(1,4): product=C.op('mul',product,rows[i][p[i]])
        odd=sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))%2
        total=C.op('sub' if odd else 'add',total,product)
        terms.append(dict(permutation=p,odd=bool(odd),product=product))
    return dict(value=total,terms=terms,counts=dict(C.n),observed_bits=C.bits)

def witness(rays):
    R=[tuple(map(F,r)) for r in rays]
    if not R or any(len(r)!=4 for r in R): raise ValueError('nonempty four-dimensional ray rows required')
    C=Counts();basis=[];pivots=[];selected=[];trace=[]
    for index,row in enumerate(R):
        z=list(row)
        for p,b in zip(pivots,basis,strict=True):
            coefficient=z[p]
            if coefficient:
                z=[C.op('sub',x,C.op('mul',coefficient,y)) for x,y in zip(z,b,strict=True)]
        pivot=next((j for j,x in enumerate(z) if x),None)
        record=dict(index=index,residual=tuple(z),pivot=pivot)
        if pivot is not None:
            denominator=z[pivot];normalized=tuple(C.op('div',x,denominator) for x in z)
            assert all(normalized[p]==0 for p in pivots)
            pivots.append(pivot);basis.append(normalized);selected.append(index);record['normalized']=normalized
        trace.append(record)
        if len(basis)==4:break
    determinant=determinant4([R[i] for i in selected]) if len(basis)==4 else None
    if determinant is not None:assert determinant['value']!=0
    return dict(rank=len(basis),selected_indices=selected,pivots=pivots,trace=trace,
                determinant=determinant,status='full_rank_witness' if determinant is not None else 'unknown_rank_deficient',
                elimination_counts=dict(C.n),elimination_observed_bits=C.bits,
                total_ray_rows=len(R),considered_ray_rows=len(trace),rational_decodes=4*len(R))

def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x
