"""Frozen exact-interval controls plus ordinary independent scalar quadrature."""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import time
import mpmath as mp
from scipy.integrate import quad
from rational_bounds import I,pi_interval,atan_point,sqrt_point,event_bounds,serial,accounting

HERE=Path(__file__).resolve().parent
OUT=HERE/'controls.json'
assert not OUT.exists(),'Preserve earlier controls'
t0=time.process_time()
mp.mp.dps=80

def m(q):return mp.mpf(q.numerator)/q.denominator
def check(interval,reference):
    assert m(interval.lo)<=reference<=m(interval.hi)

pi=pi_interval();check(pi,mp.pi)
assert F(333,106)<pi.lo<pi.hi<F(22,7)
roots=[]
for x in [F(0),F(1,4),F(2),F(9,4),F(10**12),F(1,10**12)]:
    got=sqrt_point(x);check(got,mp.sqrt(m(x)))
    assert got.lo*got.lo<=x<=got.hi*got.hi
    roots.append({'input':x,'interval':got})
angles=[]
for x in [F(0),F(1,2),F(-1,2),F(1),F(-1),F(2),F(-2),F(100),F(-100)]:
    got=atan_point(x);check(got,mp.atan(m(x)))
    angles.append({'input':x,'interval':got})
identity=atan_point(F(1,2))+atan_point(F(1,3))-pi/4
assert identity.lo<=0<=identity.hi

def integral(k):
    # erf avoids subtracting two nearly equal CDF values in the prefix factor.
    rt=math.sqrt(2);phi0=1/math.sqrt(2*math.pi)
    def fn(v):
        cdf=(1+math.erf(v/rt))/2
        return .25*phi0*math.exp(-v*v/2)*math.erf(v/(k*rt))*(cdf*cdf-.25)
    return quad(fn,0,math.inf,epsabs=2e-14,epsrel=2e-12,limit=100)

events=[]
for k in [1,2,10,100]:
    b=event_bounds(F(k));p,e=integral(k)
    assert float(b['event_probability'].lo)<=p<=float(b['event_probability'].hi)
    assert p<=float(b['rank_factor_wedge_upper'].hi)
    if k==1:
        q=F(11,384);assert b['event_probability'].lo<=q<=b['event_probability'].hi
        assert abs(p-float(q))<5e-15
    events.append({'bounds':b,'scalar_integral':p,'quad_error_estimate_uncertified':e})
out={'pi':pi,'roots':roots,'angles':angles,'addition_identity_zero_interval':identity,'event_controls':events,
     'accounting':accounting(),'cpu_seconds':time.process_time()-t0,
     'source_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['rational_bounds.py','controls.py','PROTOCOL.md']},
     'new_model_forward_MC_calls':0,'scalar_reference_integrals':4,
     'scope':'Rational bounds certified by source theorem; high-precision and adaptive numerical references are controls, not outward quadrature certificates.'}
with OUT.open('x') as f:json.dump(serial(out),f,indent=2);f.write('\n')
print(json.dumps(serial({'event_controls':events,'accounting':accounting(),'cpu_seconds':out['cpu_seconds']}),indent=2))
