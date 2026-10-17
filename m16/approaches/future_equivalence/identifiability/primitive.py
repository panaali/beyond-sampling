"""Exact sufficient global margin-state identifiability conditions."""
from fractions import Fraction as F
from collections import Counter

class Ledger:
    def __init__(self):self.counts=Counter();self.bits=[0,0]
    def op(self,kind,a,b=None):
        self.counts[kind]+=1
        if kind=='add':z=a+b
        elif kind=='sub':z=a-b
        elif kind=='mul':z=a*b
        elif kind=='div':z=a/b
        elif kind=='abs':z=abs(a)
        else:raise ValueError(kind)
        self.bits=[max(self.bits[0],abs(z.numerator).bit_length()),max(self.bits[1],z.denominator.bit_length())]
        return z

def conditions(Wi,Wh,Wo):
    H=len(Wi);assert 1<=H<=8 and len(Wh)==H and all(len(r)==H for r in Wh)
    assert len(Wo)>=2 and all(len(r)==H for r in Wo)
    Wi=list(map(F,Wi));Wh=[list(map(F,r)) for r in Wh];Wo=[list(map(F,r)) for r in Wo];K=len(Wo)-1;L=Ledger()
    A=[r[:] for r in Wh];rank=0;pivots=[];swaps=[]
    for col in range(H):
        pivot=next((i for i in range(rank,H) if A[i][col]),None)
        if pivot is None:continue
        if pivot!=rank:A[rank],A[pivot]=A[pivot],A[rank];swaps.append([rank,pivot])
        z=A[rank][col];A[rank]=[L.op('div',x,z) for x in A[rank]]
        for i in range(H):
            if i==rank or not A[i][col]:continue
            c=A[i][col];A[i]=[L.op('sub',x,L.op('mul',c,y)) for x,y in zip(A[i],A[rank],strict=True)]
        pivots.append(col);rank+=1
    rank_counts=dict(L.counts);C=[]
    for i in range(H):
        w=L.op('abs',Wi[i]);C.append(tuple(L.op('mul',w,L.op('sub',Wo[k+1][i],Wo[0][i])) for k in range(K)))
    sums=[(F(0),)*K];groups={sums[0]:[0]}
    for mask in range(1,1<<H):
        bit=mask&-mask;i=bit.bit_length()-1;previous=mask^bit
        value=tuple(L.op('add',a,b) for a,b in zip(sums[previous],C[i],strict=True))
        sums.append(value);groups.setdefault(value,[]).append(mask)
    checks=dict(nonzero_input=all(Wi),recurrent_invertible=rank==H,subset_injective=len(groups)==1<<H)
    return dict(H=H,K=K,model=[Wi,Wh,Wo],rref=A,rank=rank,pivots=pivots,swaps=swaps,
                fingerprints=C,subset_sums=sums,collision_groups=[m for m in groups.values() if len(m)>1],
                checks=checks,identifiable=all(checks.values()),rank_counts=rank_counts,counts=dict(L.counts),observed_bits=L.bits,
                scope='Sufficient real-arithmetic global margin/projective identifiability only; failed conditions are inconclusive.')

def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x
