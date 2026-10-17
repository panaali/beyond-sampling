"""Independent adaptive scalar/inner integration of selected-last pair events."""
from pathlib import Path
from fractions import Fraction as F
import math,json,time
from scipy.integrate import quad
from scipy.special import ndtr
start=time.process_time()
kappa=float(F(2048317190555034970587,5501109191530973757440));m=6
phi=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
def first(u):
    cut=u/kappa;q=float(ndtr(-u))
    p0=quad(lambda x:phi(x)*(float(ndtr(-kappa*x))-q),0,cut,epsabs=2e-14,epsrel=2e-12)[0]
    p1=q*(float(ndtr(cut))-.5)+quad(lambda x:phi(x)*float(ndtr(-kappa*x)),cut,math.inf,epsabs=2e-14,epsrel=2e-12)[0]
    return p0,p1
rows=[]
for sign,k in [('+',12170831/67108864),('-',float(F(2112615762431175989315,12339274392864452247552)))]:
    def fun(u):
        f,q=float(ndtr(u)),float(ndtr(-u));p0,p1=first(u)
        common0=f**m*p0;common1=m*f**(m-1)*q*p0+f**m*p1
        cut=u/k
        below=float(ndtr(min(u,cut) if sign=='+' else cut))-.5
        above=float(ndtr(-u)-ndtr(-cut)) if sign=='+' and cut>u else 0.
        return phi(u)*(common1*below+common0*above)
    pieces=[quad(fun,a,b,epsabs=2e-13,epsrel=2e-11) for a,b in [(0,.5),(.5,1),(1,3),(3,8)]]
    rows.append({'C_sign':sign,'threshold':k,'selected_last_joint':sum(v for v,e in pieces),'outer_adaptive_error_estimate':sum(e for v,e in pieces),
                 'event_probability':math.atan2(1,kappa)*math.atan2(1,k)/(4*math.pi**2)})
out={'rows':rows,'sum_selected_joint':sum(r['selected_last_joint'] for r in rows),'cpu_seconds':time.process_time()-start,
     'scope':'independent scalar integration; numerical estimates not certified, no author function imported'}
Path(__file__).with_name('two_pair_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
