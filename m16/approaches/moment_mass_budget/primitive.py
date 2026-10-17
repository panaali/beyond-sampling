"""Additional Dirichlet moment interval; T130 finite positive fallback preserved."""
import math
import numpy as np
from ..positive_mass_budget.primitive import Counter,C,dot,determinant,Rules

def add(xs):
    v=xs[0]
    for x in xs[1:]:v=v+x
    return v

def full_geometry(R,op):
    A=[[C(x,op) for x in row] for row in R];c=[A[0][j]+A[1][j]+A[2][j]+A[3][j] for j in range(4)];r=[dot(row,c) for row in A];m=min(r);d=[dot(row,row) for row in A];b=max(d);cc=dot(c,c);det=abs(determinant(A))
    if m<=0 or cc<=0 or det<=0:raise ArithmeticError('No positive center/determinant certificate in floating arithmetic')
    a=m*m/cc
    if b<a:raise ArithmeticError('Floating radial interval reversed')
    return a,b,det,(A,cc,r,d)

def geometry(R,op):return full_geometry(R,op)[:3]

def moments(data):
    A,s,r,d=data;t=add(d);off=[dot(A[i],A[j]) for i in range(4) for j in range(i+1,4)];d2=add([x*x for x in d]);r2=add([x*x for x in r]);off2=add([x*x for x in off]);g2=d2+2*off2;dr=dot(d,r);mu=(s+t)/20;eq2=(s*s+2*s*t+4*r2+t*t+2*g2+8*dr+6*d2)/840;var=eq2-mu*mu
    return mu,var,eq2

def estimate(R,tol,rules):
    op=Counter();out=dict(trials=[],kernel_arithmetic=0)
    try:
        a,b,det,data=full_geometry(R,op);D=det/(12*math.pi**2);lo=D/(b*b);hi=D/(a*a);mid=(lo+hi)/2;err=(hi-lo)/2;out.update(a=a.v,b=b.v,det=det.v,interval=[lo.v,hi.v])
        if err<=tol:out.update(value=mid.v,bound=err.v,mode='interval')
        else:
            mu,var,eq2=moments(data);out['moment']=dict(mean=mu.v,variance=var.v,second=eq2.v)
            if var<=0 or mu<=0:out['moment']['failure']='Nonpositive floating variance/mean; shortcut not used'
            else:
                lower=D/(mu*mu);a2=a*a;upper=lower+3*D*var/(a2*a2);ml=max(lo,lower);mh=min(hi,upper)
                if mh<ml:out['moment']['failure']='Floating interval intersection reversed; shortcut not used'
                else:
                    mm=(ml+mh)/2;me=(mh-ml)/2;out['moment'].update(interval=[ml.v,mh.v],bound=me.v)
                    if me<=tol:out.update(value=mm.v,bound=me.v,mode='moment')
            if 'value' not in out:
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
