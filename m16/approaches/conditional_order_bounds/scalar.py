"""Real-theory formulas evaluated at finite MP precision; no outward guarantee."""
import mpmath as mp

def finite(x):
 y=mp.mpf(x)
 if not mp.isfinite(y):raise ValueError('finite scalar required')
 return y
def phi(x):return mp.exp(-x*x/2)/mp.sqrt(2*mp.pi)
def Phi(x):return mp.erfc(-x/mp.sqrt(2))/2
def absolute_mgf(bound,anchor,rate):
 b,a,s=map(finite,(bound,anchor,rate))
 if s<0:raise ValueError('nonnegative rate required')
 if not s:return mp.mpf(1)
 if a>b:return mp.exp(s*a+s*s/2)*Phi(b+s)/Phi(b)
 return mp.exp(s*s/2)*(mp.exp(s*a)*Phi(a+s)+mp.exp(-s*a)*(Phi(b-s)-Phi(a-s)))/Phi(b)
def absolute_mean(bound,anchor):
 b,a=map(finite,(bound,anchor))
 if a>b:return a+phi(b)/Phi(b)
 return (2*phi(a)-phi(b)+a*(2*Phi(a)-Phi(b)))/Phi(b)
def parameters(bound,anchors,weights):
 b=finite(bound);aa=list(map(finite,anchors));ww=list(map(finite,weights))
 if len(aa)!=len(ww) or any(w<0 for w in ww):raise ValueError('matched anchors and nonnegative weights required')
 return b,aa,ww
def expected_allowance(bound,anchors,weights):
 b,aa,ww=parameters(bound,anchors,weights)
 return sum((w*absolute_mean(b,a) for a,w in zip(aa,ww)),mp.mpf(0))
def bound_at_rate(bound,anchors,weights,margin,rate):
 b,aa,ww=parameters(bound,anchors,weights);m,s=map(finite,(margin,rate))
 if s<0:raise ValueError('nonnegative rate required')
 if m<=0:return mp.mpf(1)
 if not any(ww):return mp.mpf(0)
 logb=-s*m+sum((mp.log(absolute_mgf(b,a,s*w)) for a,w in zip(aa,ww) if w),mp.mpf(0))
 return mp.exp(min(mp.mpf(0),logb))
