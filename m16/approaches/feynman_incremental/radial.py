"""Tube radial measure on the pinned corrected gate walker; no shared edits."""
import inspect,math
import numpy as np
from scipy.special import betainc,beta
from . import native

# Retain the original tube source literal, including its floating precision.
C_RAD=math.sqrt(2)*24/11.631728396567448
A=math.sqrt(2)*C_RAD
HALF_BETA=.5*A*float(beta(4.5,.5))

def radial_interval(lo,hi):
    lo=np.asarray(lo,dtype=float);hi=np.asarray(hi,dtype=float)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError('Radial interval endpoints must be finite')
    delta=np.maximum(hi-lo,0.)
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        q=1+lo*lo+lo*delta+.25*delta*delta
        midpoint=A*delta/q**5
        low2=np.where((lo<=0)&(hi>=0),0.,np.minimum(lo*lo,hi*hi))
        high2=np.maximum(lo*lo,hi*hi)
        derivative=np.minimum(10*A,10*A*(11*high2+1)/(1+low2)**7)
        bound=delta*delta*delta*derivative/24
        use=(delta>0)&np.isfinite(midpoint)&(bound<=np.finfo(float).eps*midpoint)
        reflect=hi<=0
        a=np.where(reflect,-hi,lo);b=np.where(reflect,-lo,hi)
        same=a>=0
        a2=a*a;b2=b*b
        za=np.where(same,1/(1+a2),a2/(1+a2));zb=np.where(same,1/(1+b2),b2/(1+b2))
        alpha=np.where(same,4.5,.5);beta_arg=np.where(same,.5,4.5)
        ia=betainc(alpha,beta_arg,za);ib=betainc(alpha,beta_arg,zb)
        plain=HALF_BETA*np.where(same,ia-ib,np.sign(b)*ib-np.sign(a)*ia)
        value=np.where(delta>0,np.where(use,midpoint,plain),0.)
    return value,np.where(use,bound,0.),use

# Reuse the native interval adapter exactly, replacing only its mass callback
# and ledger labels/allowances. The gate walker still comes from pinned CORE.
_source=inspect.getsource(native.walk_mass)
for old,new in (
    ("method='interval_native'","method='radial_interval_native'"),
    ('setup=500+40*n','setup=1000+40*n'),
    ('cdf_evaluations=0,','beta_evaluations=0,'),
    ('exp_evaluations=0,','radial_density_evaluations=0,'),
    ("counters['cdf_evaluations']+=2*len(lo)","counters['beta_evaluations']+=2*len(lo)"),
    ("counters['exp_evaluations']+=2*len(lo)","counters['radial_density_evaluations']+=2*len(lo)"),
    ('per_row[ids]+=940  # 3*300 interval allowance +40 assembly.',
     'per_row[ids]+=1090  # 3*350 radial interval allowance +40 assembly.')):
    if _source.count(old)!=1:raise RuntimeError('Native interval adapter source layout changed')
    _source=_source.replace(old,new)
_namespace=dict(native.__dict__);_namespace['normal_interval']=radial_interval
exec(compile(_source,__file__,'exec'),_namespace)
walk_mass=_namespace['walk_mass']
