"""Independent high-precision mass for the saved adjacent-float CDF failure."""
import json,math,time
from pathlib import Path
import mpmath as mp
from scipy.special import ndtr
start=time.process_time();mp.mp.dps=160
lo=-.940093382017101;hi=-.9400933820171009;delta=hi-lo
a,b=mp.mpf(lo),mp.mpf(hi)
truth=mp.quad(lambda x:mp.exp(-x*x/2)/mp.sqrt(2*mp.pi),[a,b])
value=delta*math.exp(-lo*lo/2-lo*delta/2-delta*delta/8)/math.sqrt(2*math.pi)
phi=lambda x:mp.exp(-x*x/2)/mp.sqrt(2*mp.pi)
lower=(b-a)*min(phi(a),phi(b));upper=(b-a)*phi(max(a,min(b,mp.mpf(0))))
assert lower<=truth<=upper and value>0
out=dict(lo_hex=lo.hex(),hi_hex=hi.hex(),ordinary_cdf_mass=float(ndtr(hi)-ndtr(lo)),
         high_precision_mass=str(truth),positive_midpoint_formula=value,
         relative_formula_error=float(abs(mp.mpf(value)/truth-1)),
         exact_endpoint_bound_width=str(upper-lower),cpu_seconds=time.process_time()-start,
         scope='Single exact-stored-float interval; high precision reference, not production outward arithmetic certification.')
Path(__file__).with_name('adjacent_cdf_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
