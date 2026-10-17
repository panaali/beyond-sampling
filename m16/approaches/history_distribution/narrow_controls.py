"""High-precision positive integration on a fixed narrow-interval set."""
from collections import defaultdict
import json,time
from pathlib import Path
import mpmath as mp
import numpy as np
from coupling import normal_mass

started=time.process_time();stats=defaultdict(float)
cases=[(-.940093382017101,-.9400933820171009)]
for center in (0.,1.,-1.,8.,-8.,30.,-30.):
    cases.append((center,np.nextafter(center,np.inf)))
    cases.append((center,center+1e-8/(1+abs(center))))
rows=[]
with mp.workdps(100):
    for a,b in cases:
        result=normal_mass(np.array([a]),np.array([b]),stats)[0]
        aa,bb=mp.mpf(float(a)),mp.mpf(float(b));width=bb-aa
        truth=width*mp.quad(lambda t:mp.exp(-(aa+width*t)**2/2)/mp.sqrt(2*mp.pi),[0,1])
        error=abs(mp.mpf(float(result))-truth)
        relative=float(error/truth) if truth else 0.
        # The smallest subnormal interval may have unrepresentable mass.
        if float(truth)>0:assert relative<1e-12,(a,b,result,str(truth),relative)
        rows.append(dict(a=float(a),b=float(b),result=float(result),truth=mp.nstr(truth,35),relative_error=relative,underflow_to_zero=bool(result==0 and truth>0)))
out=dict(rows=rows,work=dict(stats),cpu_seconds=time.process_time()-started)
(Path(__file__).parent/'narrow_controls_results.json').write_text(json.dumps(out,indent=2)+'\n')
print('narrow controls',len(rows),'CPU',out['cpu_seconds'],'max representable relative',max(x['relative_error'] for x in rows if not x['underflow_to_zero']))
