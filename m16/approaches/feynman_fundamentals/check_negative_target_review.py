"""Independent selected-negative-target integral, using scalar erfc only."""
import json,math,time
from pathlib import Path
from fractions import Fraction as F
from scipy.integrate import quad
start=time.process_time();kappa=float(F(2048317190555034970587,5501109191530973757440))
sf=lambda x:.5*math.erfc(x/math.sqrt(2))
phi=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
rows=[]
for m,k in [(0,0.),(2,0.),(6,0.),(6,kappa)]:
    pieces=[quad(lambda s:phi(s)*sf(k*s)*sf(s)**(m+2),a,b,epsabs=2e-17,epsrel=2e-13) for a,b in[(0,.25),(.25,1),(1,3),(3,8)]]
    value=sum(v for v,e in pieces);exact=F(1,2**(m+4)*(m+3)) if k==0 else None
    if exact is not None:assert abs(value-float(exact))<1e-16
    rows.append({'m':m,'kappa':k,'selected_negative_A_joint':value,'adaptive_error_estimate':sum(e for v,e in pieces),'zero_kappa_exact':str(exact) if exact is not None else None})
out={'rows':rows,'target_tail_bound':sf(8),'cpu_seconds':time.process_time()-start,
     'scope':'kneg0 only, independent scalar erfc integration; no certified quadrature bound'}
Path(__file__).with_name('negative_target_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
