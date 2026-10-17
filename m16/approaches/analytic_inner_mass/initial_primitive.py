"""Positive analytic z integration with counted array expressions and GL fallback."""
import math,time
from collections import Counter
import numpy as np
from numpy.polynomial.legendre import leggauss
# Universal constants, independent of rays/model, declared precomputed.
SCALE=1/(2*math.pi**2)
SERIES=(1/6,-1/120,1/5040,-1/362880,1/39916800)
REMAINDER=1/math.factorial(13)
class Ledger:
    def __init__(self):self.counts=Counter()
    def apply(self,name,*args,axis=None):
        if name=='sum':
            out=np.sum(args[0],axis=axis);self.counts['arithmetic']+=np.size(args[0])-np.size(out);return out
        fn={'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide,'sqrt':np.sqrt,'sin':np.sin,'atan2':np.arctan2,'lt':np.less,'le':np.less_equal,'gt':np.greater,'eq':np.equal,'finite':np.isfinite,'and':np.logical_and,'or':np.logical_or}[name]
        out=fn(*args);kind='arithmetic' if name in ('add','sub','mul','div') else 'comparison' if name in ('lt','le','gt','eq','finite','and','or') else name
        self.counts[kind]+=np.size(out);return out
    def __getattr__(self,name):return lambda *a,**kw:self.apply(name,*a,**kw)
    def dot(self,a,b):return self.sum(self.mul(a,b),axis=-1)
    def square(self,a):return self.mul(a,a)
    def dict(self):return dict(self.counts)
class Rules:
    def __init__(self):self.cache={};self.events=[];self.ledger=Ledger()
    def get(self,n):
        if n not in self.cache:
            start=time.process_time();L=self.ledger;before=L.counts.copy();x,w=leggauss(n);x=L.div(L.add(x,1),2);w=L.div(w,2);u,v=np.meshgrid(x,x,indexing='ij');wu,wv=np.meshgrid(w,w,indexing='ij');u=u.ravel();v=v.ravel();om=L.sub(1,u);vweight=L.mul(om,v);zweight=L.mul(om,L.sub(1,v));outerw=L.mul(L.mul(wu.ravel(),wv.ravel()),L.mul(om,zweight));self.cache[n]=(u,vweight,zweight,outerw,x,w);self.events.append(dict(order=n,cpu=time.process_time()-start,counts=dict(L.counts-before),legendre_internal_cost_unknown=True))
        return self.cache[n]
def positive_inner(P,D,n,x,w,L):
    """Return integral values, series-only error bounds and diagnostic mode counts."""
    P=np.asarray(P);D=np.asarray(D);Q=L.add(P,D);a=L.dot(D,D);b=L.dot(P,D);c=L.dot(P,P);e=L.dot(Q,Q)
    wedges=np.stack([L.sub(L.mul(P[:,i],D[:,j]),L.mul(P[:,j],D[:,i])) for i in range(4) for j in range(i+1,4)],axis=1);delta=L.dot(wedges,wedges)
    values=np.full(len(P),np.nan);errors=np.zeros(len(P));modes=np.full(len(P),'fallback',dtype='<U12');constant=L.apply('and',L.eq(a,0),L.gt(c,0))
    if np.any(constant):values[constant]=L.div(1,L.square(c[constant]));modes[constant]='constant'
    ac=L.mul(a,c)
    with np.errstate(all='ignore'):
        quality=L.div(delta,ac);valid=L.apply('and',L.gt(ac,0),L.apply('and',L.finite(quality),L.gt(quality,1e-12)))
    ix=np.flatnonzero(valid)
    if len(ix):
        aa,bb,cc,ee,dd=a[ix],b[ix],c[ix],e[ix],delta[ix]
        with np.errstate(all='ignore'):
            rd=L.sqrt(dd);den0=L.sqrt(L.mul(aa,cc));den1=L.sqrt(L.mul(aa,ee));angle=L.atan2(rd,L.add(cc,bb));sd=L.div(rd,L.sqrt(L.mul(cc,ee)));ct0=L.div(rd,den0);ct1=L.div(rd,den1);st0=L.div(bb,den0);st1=L.div(L.add(aa,bb),den1);onecos=L.div(L.add(L.square(L.add(ct0,ct1)),L.square(L.sub(st0,st1))),2)
            small=L.le(angle,.125);diff=np.empty(len(ix));innererror=np.zeros(len(ix))
            if np.any(small):
                z=angle[small];z2=L.square(z);poly=np.full(len(z),SERIES[-1])
                for coeff in reversed(SERIES[:-1]):poly=L.add(coeff,L.mul(z2,poly))
                z3=L.mul(z2,z);diff[small]=L.mul(z3,poly);zpower=z3
                for k in range(5):zpower=L.mul(zpower,z2)
                innererror[small]=L.mul(zpower,REMAINDER)
            if np.any(~small):diff[~small]=L.sub(angle[~small],L.sin(angle[~small]))
            amplitude=L.div(aa,L.mul(2,L.mul(dd,rd)));out=L.mul(amplitude,L.add(diff,L.mul(sd,onecos)));err=L.mul(amplitude,innererror)
            accept=L.apply('and',L.finite(out),L.apply('and',L.gt(out,0),L.apply('and',L.gt(angle,0),L.lt(angle,math.pi))))
        good=ix[accept];values[good]=out[accept];errors[good]=err[accept];modes[good]='analytic'
    fallback=np.flatnonzero(modes=='fallback')
    if len(fallback):
        points=L.add(P[fallback,None,:],L.mul(x[None,:,None],D[fallback,None,:]));q=L.dot(points,points);inner=L.div(1,L.square(q));values[fallback]=L.sum(L.mul(inner,w[None,:]),axis=1)
    if not np.all(L.apply('and',L.finite(values),L.gt(values,0))):raise ArithmeticError('Positive inner evaluation failed')
    return values,errors,dict(Counter(modes.tolist()))
def integrate(R,n,det,rules):
    L=Ledger();u,vw,zw,weight,x,w=rules.get(n);R=np.asarray(R);P=L.add(L.add(L.mul(u[:,None],R[0]),L.mul(vw[:,None],R[1])),L.mul(zw[:,None],R[3]));D=L.mul(zw[:,None],L.sub(R[2],R[3]));values,bounds,modes=positive_inner(P,D,n,x,w,L);L.counts['abs']+=1;scale=L.mul(abs(det),SCALE);value=float(L.mul(scale,L.sum(L.mul(weight,values))));bound=float(L.mul(scale,L.sum(L.mul(weight,bounds))));return dict(value=value,series_error_bound=bound,kernel_counts=L.dict(),inner_modes=modes,order=n,quadrature_dimension=2,analytic_float_error_certified=False)
