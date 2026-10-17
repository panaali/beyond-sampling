"""Independent analytic bound, not adaptive quadrature, for tiny sin^6 arc."""
import json,time
from pathlib import Path
import mpmath as mp
start=time.process_time();mp.mp.dps=180
theta=mp.mpf(float(1e-12))
upper=theta**7/7;lower=upper-theta**9/9
old=mp.mpf('1.42857141061152056e-85')
assert old<lower
out=dict(theta_exact=mp.nstr(theta,180),lower=mp.nstr(lower,180),upper=mp.nstr(upper,180),
         old_reference=str(old),old_outside_bound=True,relative_width=mp.nstr((upper-lower)/upper,30),
         cpu_seconds=time.process_time()-start,
         proof='sin(x)>=x*(1-x^2/6); (1-x^2/6)^6>=1-x^2 on[0,1]; integrate x^6-x^8 and x^6.')
Path(__file__).with_name('arc_oracle_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
