"""Counted direct scalar translation of pinned DLASQ5; index/control work separate."""
from collections import Counter
class N:
    counts=Counter()
    def __init__(self,x):self.x=float(x)
    @staticmethod
    def v(x):return x.x if isinstance(x,N) else float(x)
    def op(self,o,fn,k):N.counts[k]+=1;return N(fn(self.x,N.v(o)))
    def __add__(self,o):return self.op(o,lambda a,b:a+b,'arithmetic')
    def __sub__(self,o):return self.op(o,lambda a,b:a-b,'arithmetic')
    def __mul__(self,o):return self.op(o,lambda a,b:a*b,'arithmetic')
    def __truediv__(self,o):return self.op(o,lambda a,b:a/b,'arithmetic')
    def __neg__(self):N.counts['negation']+=1;return N(-self.x)
    def __lt__(self,o):N.counts['comparison']+=1;return self.x<N.v(o)
    def __ne__(self,o):N.counts['comparison']+=1;return self.x!=N.v(o)

def q5(z,i0,n0,pp,tau,sigma,ieee,eps):
    N.counts=Counter();z=[N(0)]+[N(v) for v in z];tau,sigma,eps=map(N,(tau,sigma,eps));zero=N(0);half=N(.5)
    out={k:N(-99) for k in ('dmin','dmin1','dmin2','dn','dnm1','dnm2')}
    def result():return dict(z=[v.x for v in z[1:]],tau=tau.x,**{k:v.x for k,v in out.items()},counts=dict(N.counts))
    if n0-i0-1<=0:return result()
    dthresh=eps*(sigma+tau)
    if tau<dthresh*half:tau=zero
    shifted=tau!=zero;j4=4*i0+pp-3;emin=z[j4+4];d=z[j4]-tau;out['dmin']=d;out['dmin1']=-z[j4]
    for j4 in range(4*i0,4*(n0-3)+1,4):
        ix=j4-2 if pp==0 else j4-3;ei=j4-1 if pp==0 else j4;qi=j4+1 if pp==0 else j4+2;oi=j4 if pp==0 else j4-1
        z[ix]=d+z[ei]
        if ieee:
            temp=z[qi]/z[ix];d=d*temp-tau
            if not shifted and d<dthresh:d=zero
            out['dmin']=min(out['dmin'],d);z[oi]=z[ei]*temp;emin=min(z[oi],emin)
        else:
            if d<zero:return result()
            z[oi]=z[qi]*(z[ei]/z[ix]);d=z[qi]*(d/z[ix])-tau
            if not shifted and d<dthresh:d=zero
            out['dmin']=min(out['dmin'],d);emin=min(emin,z[oi])
    out['dnm2']=d;out['dmin2']=out['dmin'];j4=4*(n0-2)-pp;j4p2=j4+2*pp-1
    z[j4-2]=out['dnm2']+z[j4p2]
    if not ieee and out['dnm2']<zero:return result()
    z[j4]=z[j4p2+2]*(z[j4p2]/z[j4-2]);out['dnm1']=z[j4p2+2]*(out['dnm2']/z[j4-2])-tau;out['dmin']=min(out['dmin'],out['dnm1'])
    out['dmin1']=out['dmin'];j4+=4;j4p2=j4+2*pp-1;z[j4-2]=out['dnm1']+z[j4p2]
    if not ieee and out['dnm1']<zero:return result()
    z[j4]=z[j4p2+2]*(z[j4p2]/z[j4-2]);out['dn']=z[j4p2+2]*(out['dnm1']/z[j4-2])-tau;out['dmin']=min(out['dmin'],out['dn'])
    z[j4+2]=out['dn'];z[4*n0-pp]=emin
    return result()

def envelope(length,ieee=False):
    assert length>=3
    return dict(arithmetic=(5*length+1 if ieee else 6*length-2),negation=1,comparison=(3*length-5 if ieee else 4*length-6))
