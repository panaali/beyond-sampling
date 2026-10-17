import json,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import mpmath as mp
from binary import Binary
HERE=Path(__file__).parent;start=time.process_time();out={'controls':[],'t6_sums':[]}
fixtures=[[],[1.],list(np.linspace(.1,1,16)),list(np.linspace(.1,1,17)),[np.nextafter(0.,1.)]*17]
for values in fixtures:
    c=Binary();lo,hi=c.sum((values,values));truth=sum((F.from_float(float(x)) for x in values),F(0));assert F.from_float(float(lo))<=truth<=F.from_float(float(hi))
    out['controls'].append(dict(n=len(values),lo=str(float(lo)),hi=str(float(hi)),rational_reference=str(truth),events=dict(c.counts)))
c=Binary();lo,hi=c.sum(([0.,1.],[np.inf,2.]));assert lo<=1. and np.isinf(hi);out['unresolved']=[str(float(lo)),str(float(hi))]
r=json.loads((HERE/'binary_results_final.json').read_text());mp.mp.dps=180
for order in (4,8,12,16):
    rows=[v for v in r['t6'] if v['order']==order];lower=[v['lo'] for v in rows];upper=[v['hi'] for v in rows];c=Binary();lo,hi=c.sum((lower,upper))
    exactlo=sum((F.from_float(x) for x in lower),F(0));exacthi=sum((F.from_float(x) for x in upper),F(0));assert F.from_float(float(lo))<=exactlo and F.from_float(float(hi))>=exacthi
    out['t6_sums'].append(dict(order=order,lo=str(float(lo)),hi=str(float(hi)),events=dict(c.counts)))
out['cpu_seconds']=time.process_time()-start;(HERE/'sum_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
