"""Counted scalar translation of the pinned DLASQ4 source."""
import math
from collections import Counter
class N:
    counts=Counter()
    def __init__(self,x):self.x=float(x)
    @staticmethod
    def val(x):return x.x if isinstance(x,N) else float(x)
    def op(self,o,f):N.counts['arithmetic']+=1;return N(f(self.x,N.val(o)))
    def __add__(self,o):return self.op(o,lambda a,b:a+b)
    def __sub__(self,o):return self.op(o,lambda a,b:a-b)
    def __mul__(self,o):return self.op(o,lambda a,b:a*b)
    def __truediv__(self,o):return self.op(o,lambda a,b:a/b)
    def __neg__(self):N.counts['negation']+=1;return N(-self.x)
    def cmp(self,o,f):N.counts['comparison']+=1;return f(self.x,N.val(o))
    def __eq__(self,o):return self.cmp(o,lambda a,b:a==b)
    def __lt__(self,o):return self.cmp(o,lambda a,b:a<b)
    def __gt__(self,o):return self.cmp(o,lambda a,b:a>b)
    def __le__(self,o):return self.cmp(o,lambda a,b:a<=b)

def sqrt(a):N.counts['sqrt']+=1;return N(math.sqrt(a.x))

def q4(z,i0,n0,pp,n0in,dmin,dmin1,dmin2,dn,dn1,dn2,tau,ttype,g):
    N.counts=Counter();z=[N(0)]+list(map(N,z))
    dmin,dmin1,dmin2,dn,dn1,dn2,tau,g=map(N,(dmin,dmin1,dmin2,dn,dn1,dn2,tau,g))
    zero,one,two,half,quarter,third,hundred,c1,c2,c3=map(N,(0,1,2,.5,.25,.333,100,.563,1.01,1.05))
    def finish():return dict(tau=tau.x,ttype=ttype,g=g.x,counts=dict(N.counts))
    if dmin<=zero:
        tau=-dmin;ttype=-1;return finish()
    nn=4*n0+pp
    if n0in==n0:
        if dmin==dn or dmin==dn1:
            b1=sqrt(z[nn-3])*sqrt(z[nn-5]);b2=sqrt(z[nn-7])*sqrt(z[nn-9]);a2=z[nn-7]+z[nn-5]
            if dmin==dn and dmin1==dn1:
                gap2=dmin2-a2-dmin2*quarter
                gap1=a2-dn-(b2/gap2)*b2 if gap2>zero and gap2>b2 else a2-dn-(b1+b2)
                if gap1>zero and gap1>b1:
                    s=max(dn-(b1/gap1)*b1,half*dmin);ttype=-2
                else:
                    s=zero
                    if dn>b1:s=dn-b1
                    if a2>(b1+b2):s=min(s,a2-(b1+b2))
                    s=max(s,third*dmin);ttype=-3
            else:
                ttype=-4;s=quarter*dmin
                if dmin==dn:
                    gam=dn;a2=zero
                    if z[nn-5]>z[nn-7]:return finish()
                    b2=z[nn-5]/z[nn-7];np_=nn-9
                else:
                    np_=nn-2*pp;gam=dn1
                    if z[np_-4]>z[np_-2]:return finish()
                    a2=z[np_-4]/z[np_-2]
                    if z[nn-9]>z[nn-11]:return finish()
                    b2=z[nn-9]/z[nn-11];np_=nn-13
                a2=a2+b2
                for i4 in range(np_,4*i0-2+pp,-4):
                    if b2==zero:break
                    b1=b2
                    if z[i4]>z[i4-2]:return finish()
                    b2=b2*(z[i4]/z[i4-2]);a2=a2+b2
                    if hundred*max(b2,b1)<a2 or c1<a2:break
                a2=c3*a2
                if a2<c1:s=gam*(one-sqrt(a2))/(one+a2)
        elif dmin==dn2:
            ttype=-5;s=quarter*dmin;np_=nn-2*pp;b1=z[np_-2];b2=z[np_-6];gam=dn2
            if z[np_-8]>b2 or z[np_-4]>b1:return finish()
            a2=(z[np_-8]/b2)*(one+z[np_-4]/b1)
            if n0-i0>2:
                b2=z[nn-13]/z[nn-15];a2=a2+b2
                for i4 in range(nn-17,4*i0-2+pp,-4):
                    if b2==zero:break
                    b1=b2
                    if z[i4]>z[i4-2]:return finish()
                    b2=b2*(z[i4]/z[i4-2]);a2=a2+b2
                    if hundred*max(b2,b1)<a2 or c1<a2:break
                a2=c3*a2
            if a2<c1:s=gam*(one-sqrt(a2))/(one+a2)
        else:
            if ttype==-6:g=g+third*(one-g)
            elif ttype==-18:g=quarter*third
            else:g=quarter
            s=g*dmin;ttype=-6
    elif n0in==n0+1:
        if dmin1==dn1 and dmin2==dn2:
            ttype=-7;s=third*dmin1
            if z[nn-5]>z[nn-7]:return finish()
            b1=z[nn-5]/z[nn-7];b2=b1
            if not b2==zero:
                for i4 in range(4*n0-9+pp,4*i0-2+pp,-4):
                    a2=b1
                    if z[i4]>z[i4-2]:return finish()
                    b1=b1*(z[i4]/z[i4-2]);b2=b2+b1
                    if hundred*max(b1,a2)<b2:break
            b2=sqrt(c3*b2);a2=dmin1/(one+b2*b2);gap2=half*dmin2-a2
            if gap2>zero and gap2>b2*a2:s=max(s,a2*(one-c2*a2*(b2/gap2)*b2))
            else:s=max(s,a2*(one-c2*b2));ttype=-8
        else:
            s=quarter*dmin1
            if dmin1==dn1:s=half*dmin1
            ttype=-9
    elif n0in==n0+2:
        if dmin2==dn2 and two*z[nn-5]<z[nn-7]:
            ttype=-10;s=third*dmin2
            if z[nn-5]>z[nn-7]:return finish()
            b1=z[nn-5]/z[nn-7];b2=b1
            if not b2==zero:
                for i4 in range(4*n0-9+pp,4*i0-2+pp,-4):
                    if z[i4]>z[i4-2]:return finish()
                    b1=b1*(z[i4]/z[i4-2]);b2=b2+b1
                    if hundred*b1<b2:break
            b2=sqrt(c3*b2);a2=dmin2/(one+b2*b2)
            gap2=z[nn-7]+z[nn-9]-sqrt(z[nn-11])*sqrt(z[nn-9])-a2
            if gap2>zero and gap2>b2*a2:s=max(s,a2*(one-c2*a2*(b2/gap2)*b2))
            else:s=max(s,a2*(one-c2*b2))
        else:s=quarter*dmin2;ttype=-11
    else:
        assert n0in>n0+2;s=zero;ttype=-12
    tau=s
    return finish()
