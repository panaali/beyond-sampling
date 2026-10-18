"""Counted scalar setup, positive intervals and finite short Gaussian rules."""
import itertools,math,time
import numpy as np
from numpy.polynomial.legendre import leggauss
PERMS=[(p,(-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))) for p in itertools.permutations(range(4))]
class Counter:
    def __init__(self):self.arithmetic=0;self.comparison=0;self.sqrt=0;self.abs=0
    def dict(self):return vars(self).copy()
class C:
    def __init__(self,v,op):self.v=float(v);self.op=op
    def val(self,x):return x.v if isinstance(x,C) else float(x)
    def __add__(self,x):self.op.arithmetic+=1;return C(self.v+self.val(x),self.op)
    __radd__=__add__
    def __sub__(self,x):self.op.arithmetic+=1;return C(self.v-self.val(x),self.op)
    def __rsub__(self,x):self.op.arithmetic+=1;return C(self.val(x)-self.v,self.op)
    def __mul__(self,x):self.op.arithmetic+=1;return C(self.v*self.val(x),self.op)
    __rmul__=__mul__
    def __truediv__(self,x):self.op.arithmetic+=1;return C(self.v/self.val(x),self.op)
    def __rtruediv__(self,x):self.op.arithmetic+=1;return C(self.val(x)/self.v,self.op)
    def __lt__(self,x):self.op.comparison+=1;return self.v<self.val(x)
    def __le__(self,x):self.op.comparison+=1;return self.v<=self.val(x)
    def __abs__(self):self.op.abs+=1;return C(abs(self.v),self.op)
    def root(self):self.op.sqrt+=1;return C(math.sqrt(self.v),self.op)

def dot(a,b):
    terms=[x*y for x,y in zip(a,b)];s=terms[0]
    for v in terms[1:]:s=s+v
    return s

def determinant(A):
    # Exactly24 products, each3 multiplies;23 signed sums. No hidden LU.
    terms=[]
    for p,s in PERMS:terms.append((A[0][p[0]]*A[1][p[1]]*A[2][p[2]]*A[3][p[3]],s))
    v=terms[0][0]
    for t,s in terms[1:]:v=v+t if s>0 else v-t
    return v

def geometry(R,op):
    A=[[C(x,op) for x in row] for row in R];c=[A[0][j]+A[1][j]+A[2][j]+A[3][j] for j in range(4)];m=min(dot(row,c) for row in A);b=max(dot(row,row) for row in A);cc=dot(c,c);det=abs(determinant(A))
    if m<=0 or cc<=0 or det<=0:raise ArithmeticError('No positive center/determinant certificate in floating arithmetic')
    a=m*m/cc
    if b<a:raise ArithmeticError('Floating radial interval reversed')
    return a,b,det

class Rules:
    def __init__(self):self.cache={};self.events=[];self.node_plan_arithmetic=0
    def get(self,n):
        if n not in self.cache:
            t=time.process_time();x,w=leggauss(n);x=(x+1)/2;w=w/2;u,v,z=np.meshgrid(x,x,x,indexing='ij');a,b,c=np.meshgrid(w,w,w,indexing='ij');y=np.stack([u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z)],axis=-1).reshape(-1,4);weights=(a*b*c*(1-u)**2*(1-v)).ravel();self.cache[n]=(y,weights);self.node_plan_arithmetic+=18*n**3+3*n;self.events.append(dict(order=n,cpu=time.process_time()-t,node_plan_arithmetic=18*n**3+3*n,leggauss_arithmetic_unknown=True))
        return self.cache[n]

def estimate(R,tol,rules):
    op=Counter();out=dict(trials=[],kernel_arithmetic=0)
    try:
        a,b,det=geometry(R,op);D=det/(12*math.pi**2);lo=D/(b*b);hi=D/(a*a);mid=(lo+hi)/2;err=(hi-lo)/2;out.update(a=a.v,b=b.v,det=det.v,interval=[lo.v,hi.v])
        if err<=tol:out.update(value=mid.v,bound=err.v,mode='interval')
        else:
            alpha=(a+b)/2;delta=(a*b).root();sa=a.root();sb=b.root();rho=(sb-sa)/(sb+sa);one=1-rho;scale=det/(6*math.pi**2)
            for n in (2,3,4,6,8,12,16,20,24,32):
                power=rho
                for k in range(n-2):power=power*rho
                tail=2*power/(delta*delta*delta)*(alpha/one+delta*((n-1)/one+rho/(one*one)));bd=scale*tail;out['trials'].append(dict(order=n,bound=bd.v))
                if bd<=tol:
                    y,w=rules.get(n);points=y@R;out['value']=float(det.v/(2*math.pi**2)*np.dot(w,1/np.sum(points*points,axis=1)**2));out.update(bound=bd.v,mode='quadrature',order=n,kernel_arithmetic=39*n**3+1);break
            else:out.update(mode='unresolved',reason='Finite order set did not reach target')
    except Exception as exc:out.update(mode='unresolved',reason=type(exc).__name__+': '+str(exc))
    out['setup_selection_counts']=op.dict();return out
