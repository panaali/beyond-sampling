"""Pointed 3D Gaussian cone mass/second moment; exact geometry, finite MP values."""
from fractions import Fraction as F
from collections import Counter
from itertools import combinations
import math
import mpmath as mp

class Exact:
    def __init__(self):self.count=Counter();self.bits=[0,0]
    def op(self,k,a,b):
        self.count[k]+=1
        z={'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[k]()
        self.bits=[max(self.bits[0],abs(z.numerator).bit_length()),max(self.bits[1],z.denominator.bit_length())];return z
    def dot(self,x,y):
        z=self.op('mul',x[0],y[0])
        for a,b in zip(x[1:],y[1:]):z=self.op('add',z,self.op('mul',a,b))
        return z
    def cross(self,x,y):
        return tuple(self.op('sub',self.op('mul',x[j],y[k]),self.op('mul',x[k],y[j])) for j,k in [(1,2),(2,0),(0,1)])
    def rank(self,rows):
        a=[list(r) for r in rows];rank=0
        for col in range(3):
            pivot=next((j for j in range(rank,len(a)) if a[j][col]),None)
            if pivot is None:continue
            a[rank],a[pivot]=a[pivot],a[rank];d=a[rank][col]
            a[rank]=[self.op('div',x,d) for x in a[rank]]
            for j in range(len(a)):
                if j==rank or not a[j][col]:continue
                d=a[j][col];a[j]=[self.op('sub',x,self.op('mul',d,y)) for x,y in zip(a[j],a[rank])]
            rank+=1
            if rank==len(a):break
        return rank
    def canonical(self,x):
        d=abs(next(z for z in x if z));return tuple(self.op('div',z,d) for z in x)

def rational(x):
    if not isinstance(x,(int,float,F)):raise TypeError('Only real int/float/Fraction inputs')
    if isinstance(x,float) and not math.isfinite(x):raise ValueError('Nonfinite input')
    return F(x)
def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x

def prepare(normals):
    if not isinstance(normals,(list,tuple)) or len(normals)>4:raise ValueError('At most four normal vectors required')
    if any(not isinstance(n,(list,tuple)) or len(n)!=3 for n in normals):raise ValueError('Three-coordinate normals required')
    A=Exact();source=[tuple(map(rational,n)) for n in normals];unique=[];removed=[]
    for j,n in enumerate(source):
        if not any(n):removed.append(dict(index=j,reason='zero'));continue
        v=A.canonical(n)
        if v in unique:removed.append(dict(index=j,reason='positive_duplicate'));continue
        unique.append(v)
    c=dict(source=source,normals=unique,removed=removed)
    def finish(status):
        c.update(status=status,counts=dict(A.count),bits=A.bits,
                 limits='Named Fraction operations only. Comparisons, signs, canonical searches, sorting, GCD, conversions, transient bits, control/memory and MP/native work incompletely priced.');return c
    rank=A.rank(unique);c['normal_rank']=rank
    if rank<3:return finish('unsupported_lineality')
    # Full normal rank implies pointedness; all extreme rays meet two independent supporting planes.
    rays=[]
    for i,j in combinations(range(len(unique)),2):
        x=A.cross(unique[i],unique[j])
        if not any(x):continue
        for sign in (1,-1):
            r=tuple(A.op('mul',F(sign),z) for z in x)
            if all(A.dot(n,r)>=0 for n in unique):
                r=A.canonical(r)
                if r not in rays:rays.append(r)
    rays=sorted(rays);c['rays']=rays;c['ray_rank']=A.rank(rays)
    if c['ray_rank']<3:return finish('measure_zero')
    assert 3<=len(rays)<=4
    center=tuple(sum((r[k] for r in rays),F(0)) for k in range(3))
    # Charge the explicit sum additions, including its written zero initial value.
    A.count['add']+=3*len(rays)
    for z in center:A.bits=[max(A.bits[0],abs(z.numerator).bit_length()),max(A.bits[1],z.denominator.bit_length())]
    assert all(A.dot(n,center)>0 for n in unique)
    c['center']=center;facets=[];incidence=Counter()
    for index,n in enumerate(unique):
        on=[j for j,r in enumerate(rays) if A.dot(n,r)==0]
        if len(on)<2:continue
        assert len(on)==2 and A.rank([rays[j] for j in on])==2
        i,j=on;r,s=rays[i],rays[j]
        det=A.dot(center,A.cross(r,s))
        if det<0:i,j=j,i;r,s=s,r;det=-det;A.count['neg']+=1
        assert det>0
        rr=A.dot(r,r);ss=A.dot(s,s);rs=A.dot(r,s)
        ar=A.op('div',rs,rr);bs=A.op('div',rs,ss)
        pr=tuple(A.op('sub',y,A.op('mul',ar,x)) for x,y in zip(r,s))
        ps=tuple(A.op('sub',x,A.op('mul',bs,y)) for x,y in zip(r,s))
        assert A.dot(n,pr)==A.dot(n,ps)==0 and A.dot(r,pr)==A.dot(s,ps)==0
        assert A.dot(s,pr)>0 and A.dot(r,ps)>0
        f=dict(normal_index=index,ray_indices=[i,j],normal=n,normal_norm2=A.dot(n,n),r=r,s=s,
               tangent_r=pr,tangent_s=ps,tangent_r_norm2=A.dot(pr,pr),tangent_s_norm2=A.dot(ps,ps),
               triangle=dict(det=det,center_norm2=A.dot(center,center),r_norm2=rr,s_norm2=ss,cr=A.dot(center,r),rs=rs,sc=A.dot(s,center)))
        facets.append(f);incidence.update([i,j])
    assert len(facets)==len(rays) and all(incidence[i]==2 for i in range(len(rays)))
    c['facets']=facets;return finish('pointed_full')

def mpq(x):return mp.mpf(x.numerator)/x.denominator

def evaluate(c,dps):
    if c['status']=='unsupported_lineality':raise ValueError('Rank<3 explicitly unsupported')
    with mp.workdps(dps):
        if c['status']=='measure_zero':return dict(mass='0.0',matrix=[['0.0']*3 for _ in range(3)],triangles=[],boundary=[])
        assert c['status']=='pointed_full'
        triangles=[];mass=mp.mpf(0);boundary=mp.matrix(3,3)
        for f in c['facets']:
            t=f['triangle'];nc=mp.sqrt(mpq(t['center_norm2']));nr=mp.sqrt(mpq(t['r_norm2']));ns=mp.sqrt(mpq(t['s_norm2']))
            denominator=nc*nr*ns+mpq(t['cr'])*ns+mpq(t['rs'])*nc+mpq(t['sc'])*nr
            p=mp.atan2(mpq(t['det']),denominator)/(2*mp.pi)
            assert p>0;mass+=p;triangles.append(mp.nstr(p,dps))
            normal=[mpq(x)/mp.sqrt(mpq(f['normal_norm2'])) for x in f['normal']]
            tangent=[mpq(x)/mp.sqrt(mpq(f['tangent_r_norm2']))+mpq(y)/mp.sqrt(mpq(f['tangent_s_norm2'])) for x,y in zip(f['tangent_r'],f['tangent_s'])]
            for i in range(3):
                for j in range(3):boundary[i,j]+=normal[i]*tangent[j]/(4*mp.pi)
        M=boundary+mp.eye(3)*mass
        return dict(mass=mp.nstr(mass,dps),matrix=[[mp.nstr(M[i,j],dps) for j in range(3)] for i in range(3)],
                    triangles=triangles,boundary=[[mp.nstr(boundary[i,j],dps) for j in range(3)] for i in range(3)])
