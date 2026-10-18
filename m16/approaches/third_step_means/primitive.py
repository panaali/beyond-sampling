"""Exact stored-input cone algebra plus MP first moments, not FP certificates."""
from fractions import Fraction as F
from collections import Counter
import mpmath as mp

class Exact:
    def __init__(self):self.count=Counter();self.peak_num=0;self.peak_den=0
    def op(self,k,a,b):
        self.count[k]+=1
        z={'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b}[k]()
        self.peak_num=max(self.peak_num,abs(z.numerator).bit_length())
        self.peak_den=max(self.peak_den,z.denominator.bit_length());return z
    def dot(self,a,b):
        xs=[self.op('mul',x,y) for x,y in zip(a,b)];z=xs[0]
        for x in xs[1:]:z=self.op('add',z,x)
        return z

def mpq(x):return mp.mpf(x.numerator)/x.denominator

def prepare(normals,v):
    """At most two sector halfspaces in xy; append output halfspace v.G>=0.

    Fractions represent supplied exact dyadic values, without tolerance-based
    zero or proportionality decisions. Positive-proportional duplicates merge.
    Opposite constraints imply measure zero. Distinct redundant planes retain
    their exact zero facet probability; no sampled facet classification.
    """
    assert len(normals)<=2 and len(v)==3 and all(len(n)==3 and n[2]==0 for n in normals)
    A=Exact();v=tuple(map(F,v));source=[tuple(map(F,n)) for n in normals]+[v]
    if all(x==0 for x in v):return {'status':'zero_output','normals':[],'v':v,'counts':dict(A.count),'bits':[0,0]}
    unique=[];removed=[]
    for i,n in enumerate(source):
        nn=A.dot(n,n)
        if nn==0:removed.append({'index':i,'reason':'zero_constraint'});continue
        duplicate=False
        for j,u in enumerate(unique):
            uu=A.dot(u,u);nu=A.dot(n,u)
            gram=A.op('sub',A.op('mul',nn,uu),A.op('mul',nu,nu))
            assert gram>=0
            if gram==0:
                if nu<0:return {'status':'opposite_measure_zero','normals':unique+[n],'v':v,'counts':dict(A.count),'bits':[A.peak_num,A.peak_den]}
                duplicate=True;removed.append({'index':i,'reason':'positive_proportional','to':j});break
        if not duplicate:unique.append(n)
    G=[[A.dot(x,y) for y in unique] for x in unique]
    facets=[]
    for i,n in enumerate(unique):
        others=[j for j in range(len(unique)) if j!=i]
        row={'normal':n,'norm2':G[i][i],'v_dot_normal':A.dot(v,n),'other_count':len(others)}
        if len(others)==2:
            j,k=others
            p=A.op('sub',A.op('mul',G[j][j],G[i][i]),A.op('mul',G[i][j],G[i][j]))
            q=A.op('sub',A.op('mul',G[k][k],G[i][i]),A.op('mul',G[i][k],G[i][k]))
            b=A.op('sub',A.op('mul',G[j][k],G[i][i]),A.op('mul',G[i][j],G[i][k]))
            rad=A.op('sub',A.op('mul',p,q),A.op('mul',b,b))
            assert p>0 and q>0 and rad>=0
            row.update(projected_norm_products=[p,q],projected_dot=b,angle_radicand=rad)
        facets.append(row)
    return {'status':'prepared','normals':unique,'v':v,'removed':removed,'facets':facets,
            'counts':dict(A.count),'bits':[A.peak_num,A.peak_den],
            'count_limits':'named Fraction arithmetic only; exact comparisons/loops/gcd/conversion/MP/native not fully priced; reduced result bits only'}

def evaluate(certificate,dps):
    with mp.workdps(dps):
        if certificate['status']!='prepared':return {'value':'0.0','terms':[],'status':certificate['status']}
        terms=[];total=mp.mpf(0)
        for row in certificate['facets']:
            k=row['other_count']
            if k==0:p=mp.mpf(1)
            elif k==1:p=mp.mpf('0.5')
            else:
                # Exact nonnegative radicand and exact signed dot determine
                # the quadrant, including zero-probability redundant facets.
                angle=mp.atan2(mp.sqrt(mpq(row['angle_radicand'])),-mpq(row['projected_dot']))
                p=angle/(2*mp.pi)
            term=mpq(row['v_dot_normal'])/mp.sqrt(mpq(row['norm2']))*p/mp.sqrt(2*mp.pi)
            total+=term;terms.append({'probability':mp.nstr(p,dps),'contribution':mp.nstr(term,dps)})
        return {'value':mp.nstr(total,dps),'terms':terms,'status':'evaluated','negative':bool(total<0)}

def radial(a,b,dps):
    """Conditional Rayleigh/Gaussian mean, MP control form, no binary64 claim."""
    with mp.workdps(dps):
        a=mpq(F(a));b=abs(mpq(F(b)))
        if b==0:return mp.nstr(max(a,mp.mpf(0))*mp.sqrt(mp.pi/2),dps)
        # Deliberately retains direct MP subtraction for a<0. Fixed precision
        # and independent positive-angular reference expose any cancellation.
        return mp.nstr((b+a*(mp.pi/2+mp.atan(a/b)))/mp.sqrt(2*mp.pi),dps)

def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [pack(v) for v in x]
    return x
