"""Independent exact integer prefix cones. No estT, floating geometry or model imports."""
from dataclasses import dataclass
from math import gcd

class Ledger:
    def __init__(self):
        self.counts={k:0 for k in ('mul','add','sub','integer_div','gcd','rank_calls','clips','lifts')}
        self.max_bits=0
    def watch(self,vs):
        for v in vs:self.max_bits=max(self.max_bits,abs(v).bit_length())
    def dot(self,a,b):
        assert len(a)==len(b)
        s=0
        for x,y in zip(a,b,strict=True):
            v=x*y;s+=v;self.counts['mul']+=1;self.counts['add']+=1;self.watch((v,s))
        return s
    def primitive(self,v):
        v=tuple(v);self.watch(v);g=0
        for x in v:g=gcd(g,abs(x));self.counts['gcd']+=1
        assert g>0,'zero vector cannot be a ray or normal'
        self.counts['integer_div']+=len(v)
        return tuple(x//g for x in v)  # gcd is positive: never flip a direction
    def rank(self,rows,d):
        self.counts['rank_calls']+=1
        a=[list(r) for r in rows if any(r)];assert all(len(r)==d for r in a)
        k=0
        for col in range(d):
            j=next((j for j in range(k,len(a)) if a[j][col]),None)
            if j is None:continue
            a[k],a[j]=a[j],a[k];pivot=a[k][col]
            for j in range(k+1,len(a)):
                b=a[j][col]
                if not b:continue
                row=[pivot*x-b*y for x,y in zip(a[j],a[k],strict=True)]
                self.counts['mul']+=2*d;self.counts['sub']+=d;self.watch(row)
                a[j]=list(self.primitive(row)) if any(row) else row
            k+=1
            if k==len(a):break
        return k

@dataclass(frozen=True)
class Cone:
    rays:tuple
    normals:tuple
    @property
    def d(self):return len(self.rays[0])

def cone(rays,normals,L):
    rr=tuple(sorted(set(L.primitive(r) for r in rays)))
    nn=tuple(sorted(set(L.primitive(n) for n in normals)))
    assert rr and nn
    assert all(len(r)==len(rr[0]) for r in rr+nn)
    return Cone(rr,nn)

def clip(C,v,L):
    """Return full-dimensional ON/OFF cones only; globally zero form is OFF."""
    L.counts['clips']+=1;v=tuple(v);assert len(v)==C.d
    if not any(v):return None,C
    v=L.primitive(v);ss=[L.dot(v,r) for r in C.rays]
    pp=[i for i,s in enumerate(ss) if s>0];mm=[i for i,s in enumerate(ss) if s<0]
    assert pp or mm,'nonzero form vanishes on allegedly full-dimensional cone'
    if not pp:return None,C
    if not mm:return C,None
    zz=[C.rays[i] for i,s in enumerate(ss) if s==0]
    active=[tuple(j for j,n in enumerate(C.normals) if L.dot(n,r)==0) for r in C.rays]
    new=[];cache={}
    for i in pp:
        for j in mm:
            common=tuple(sorted(set(active[i]).intersection(active[j])))
            if common not in cache:cache[common]=L.rank([C.normals[k] for k in common],C.d)
            if cache[common]!=C.d-2:continue
            r=tuple((-ss[j])*x+ss[i]*y for x,y in zip(C.rays[i],C.rays[j],strict=True))
            L.counts['mul']+=2*C.d;L.counts['add']+=C.d;new.append(L.primitive(r))
    assert new,'a genuine full-dimensional cut has crossing edges'
    plus=cone([C.rays[i] for i in pp]+zz+new,C.normals+(v,),L)
    minus=cone([C.rays[i] for i in mm]+zz+new,C.normals+(tuple(-x for x in v),),L)
    return plus,minus

def lift(C,v,L):
    """Cut C x R by a fresh form with nonzero last coefficient."""
    L.counts['lifts']+=1;v=tuple(v);assert len(v)==C.d+1 and v[-1]!=0
    a=v[-1];sign=1 if a>0 else -1;boundary=[]
    for r in C.rays:
        z=-sign*L.dot(v[:-1],r)
        q=tuple(abs(a)*x for x in r)+(z,)
        L.counts['mul']+=C.d+1;boundary.append(q)
    n=tuple(tuple(r)+(0,) for r in C.normals)
    axis=(0,)*C.d+(sign,)
    return cone(boundary+[axis],n+(v,),L),cone(boundary+[tuple(-x for x in axis)],n+(tuple(-x for x in v),),L)

def certify_cone(C,L):
    d=C.d
    assert L.rank(C.rays,d)==d and L.rank(C.normals,d)==d
    for r in C.rays:
        vals=[L.dot(n,r) for n in C.normals];assert min(vals)>=0
        assert L.rank([n for n,s in zip(C.normals,vals,strict=True) if s==0],d)==d-1
    w=tuple(sum(r[j] for r in C.rays) for j in range(d));L.counts['add']+=len(C.rays)*d
    w=L.primitive(w);assert all(L.dot(n,w)>0 for n in C.normals)
    return w

@dataclass(frozen=True)
class State:
    C:Cone
    hidden:tuple
    history:tuple

def validate_weights(wi,wh,D):
    assert isinstance(D,int) and D>0 and wi and any(wi)
    h=len(wi);assert len(wh)==h and all(len(r)==h for r in wh)
    assert all(type(x) is int for x in wi) and all(type(x) is int for r in wh for x in r)

def initial(wi,wh,D,L):
    validate_weights(wi,wh,D);out=[]
    for sign in (-1,1):
        mask=tuple(int(sign*w>0) for w in wi)
        C=cone([(sign,)],[(sign,)],L)
        out.append(State(C,tuple((w if on else 0,) for w,on in zip(wi,mask,strict=True)),(mask,)))
    return out

def extend(S,wi,wh,D,L):
    """One source state -> every full-dimensional next-step gate history."""
    d=S.C.d;h=len(wi)
    A=[]
    for j in range(h):
        row=tuple(L.dot(wh[j],tuple(S.hidden[k][i] for k in range(h))) for i in range(d))
        fresh=wi[j]*D**d;L.counts['mul']+=1;L.watch((fresh,));A.append(row+(fresh,))
    pivot=next(i for i,w in enumerate(wi) if w)
    on,off=lift(S.C,A[pivot],L);work=[]
    for C,bit in ((on,1),(off,0)):
        mask=[None]*h;mask[pivot]=bit;work.append((C,mask))
    for j in range(h):
        if j==pivot:continue
        nxt=[]
        for C,mask in work:
            on,off=clip(C,A[j],L)
            for child,bit in ((on,1),(off,0)):
                if child is not None:
                    m=mask.copy();m[j]=bit;nxt.append((child,m))
        work=nxt
    return [State(C,tuple(row if bit else (0,)*(d+1) for row,bit in zip(A,mask,strict=True)),S.history+(tuple(mask),)) for C,mask in work]

def certify_state(S,wi,wh,D,L):
    w=certify_cone(S.C,L);d=len(w);hidden=[0]*len(wi);forms=[]
    for t,mask in enumerate(S.history):
        nxt=[]
        for j,on in enumerate(mask):
            value=L.dot(wh[j],hidden)+wi[j]*D**t*w[t]
            L.counts['mul']+=2;L.counts['add']+=1
            assert (value>0)==bool(on)
            nxt.append(value if on else 0)
        hidden=nxt
    assert tuple(hidden)==tuple(L.dot(row,w) for row in S.hidden)
    # Rebuild every complete linear gate form, distinguishing a true zero form.
    old=tuple((0,)*d for _ in wi)
    for t,mask in enumerate(S.history):
        new=[]
        for j,on in enumerate(mask):
            row=[L.dot(wh[j],tuple(old[k][q] for k in range(len(wi)))) for q in range(d)]
            row[t]+=wi[j]*D**t;L.counts['mul']+=1;L.counts['add']+=1
            if any(row):
                signed=tuple(x if on else -x for x in row);assert L.dot(signed,w)>0;forms.append(signed)
            else:assert not on
            new.append(tuple(row) if on else (0,)*d)
        old=tuple(new)
    return dict(witness=w,strict_forms=len(forms),zero_forms=len(wi)*d-len(forms))

def encode(S):
    return dict(rays=S.C.rays,normals=S.C.normals,hidden=S.hidden,history=S.history)
