"""Exact stored-matrix target, not source covariance or mass certification."""
from fractions import Fraction as F
from itertools import permutations
from collections import Counter
from frozen import rational_bounds as rb

COUNTS=Counter()
BITS=160

def det(a):
    n=len(a);assert all(len(r)==n for r in a)
    out=F(0)
    for p in permutations(range(n)):
        term=F(1)
        for i,j in enumerate(p):term=rb.calc('mul',term,a[i][j])
        inversions=sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))
        out=rb.calc('add' if inversions%2==0 else 'sub',out,term)
        COUNTS['permutation_terms']+=1
    COUNTS[f'determinant_{n}']+=1
    return out

def sub(a,rows,cols):return [[a[i][j] for j in cols] for i in rows]
def cofactor(a,k,l):
    rows=[i for i in range(4) if i!=k];cols=[i for i in range(4) if i!=l]
    d=det(sub(a,rows,cols));COUNTS['cofactor_sign']+=1
    return -d if (k+l)%2 else d

def rounded_interval(v):
    scale=1<<BITS
    lo=(v.lo.numerator*scale)//v.lo.denominator
    hi=-((-v.hi.numerator*scale)//v.hi.denominator)
    COUNTS['outward_dyadic_endpoints']+=2
    ans=rb.I(F(lo,scale),F(hi,scale));assert ans.lo<=v.lo<=v.hi<=ans.hi
    return ans

def enclose(a,pair):
    assert len(a)==4 and all(len(r)==4 for r in a)
    a=[[F(x) for x in row] for row in a]
    assert len(pair)==2 and len(set(pair))==2 and all(0<=i<4 for i in pair)
    if any(a[i][j]!=a[j][i] for i in range(4) for j in range(4)):
        return {'status':'unknown_asymmetric'}
    leading=[]
    for n in range(1,5):
        leading.append(det(sub(a,range(n),range(n))))
        if leading[-1]<=0:return {'status':'unknown_not_spd','leading':leading}
    i,j=pair;k,l=[n for n in range(4) if n not in pair]
    c=cofactor(a,k,l);ck=cofactor(a,k,k);cl=cofactor(a,l,l)
    d2=det(sub(a,pair,pair));rad=rb.calc('mul',leading[-1],d2)
    jac=rb.calc('sub',rb.calc('mul',ck,cl),rb.calc('mul',c,c))
    assert rad==jac,'Jacobi identity failed'
    out={'status':'unknown_nonpositive_radicand','leading':leading,'det2':d2,'cofactor':c,'radicand':rad,'jacobi':jac}
    if rad<=0:return out
    x=rb.sqrt_point(rad);out['denominator']=x
    if x.lo<=0:out['status']='unknown_precision';return out
    ratio=rounded_interval(rb.I.point(-c)/x)
    angle=rb.atan_interval(ratio)
    out.update(status='bounded',angle=angle,ratio=ratio)
    return out

def accounting():return {'matrix_categories':dict(COUNTS),'interval_categories':rb.accounting()}
def reset():
    COUNTS.clear();rb.COUNTS.clear();rb.PEAK[:]=[0,0];rb.pi_interval.cache_clear()
