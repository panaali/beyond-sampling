"""Declared output/domain edges, independent high-precision q^-3 control."""
import json,time
from pathlib import Path
import numpy as np
import mpmath as mp
from binary import Binary
from t6_subset import q3
HERE=Path(__file__).parent

def mf(x):
    if np.isinf(x):return mp.inf if x>0 else -mp.inf
    n,d=float(x).as_integer_ratio();return mp.mpf(n)/d
mp.mp.dps=160;start=time.process_time();out=[]
for name,a,b,det in [('constant',1.,1.,1.),('zero_determinant',.25,1.,0.),('denominator_underflow',1e-300,2e-300,1.),('product_overflow',1e300,2e300,1.)]:
    c=Binary();lo,hi=c.bound(a,b,3,det,det);truth=mf(det)*q3(mf(a),mf(b),4)/(60*mp.pi**3) if a!=b else mp.mpf(0)
    assert mf(lo)<=truth<=mf(hi),(name,lo,hi,truth)
    if name in ('constant','zero_determinant'):assert lo==hi==0
    out.append(dict(name=name,lo=str(float(lo)),hi=str(float(hi)),reference=mp.nstr(truth,170),counts=dict(c.counts)))
(HERE/'binary_edge_results.json').write_text(json.dumps(dict(cases=out,cpu_seconds=time.process_time()-start),indent=2)+'\n');print(out,flush=True)
