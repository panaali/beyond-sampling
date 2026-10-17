"""Deterministic reflected normal intervals; independent high precision erfc."""
import json,time
from pathlib import Path
import numpy as np
import mpmath as mp
from .native import normal_interval

def main():
    start=time.process_time();mp.mp.dps=100;rows=[]
    fixtures=[]
    for a in (8.,10.,20.):
        for b in (np.nextafter(a,np.inf),a+.1,a+1):
            fixtures.extend(((a,b),(-b,-a)))
    fixtures.extend(((-1.,1.),(-1e-12,1e-12),(0.,np.nextafter(0.,1.))))
    for a,b in fixtures:
        aa,bb=mp.mpf(a),mp.mpf(b)
        truth=(mp.erfc(aa/mp.sqrt(2))-mp.erfc(bb/mp.sqrt(2)))/2 if a>=0 else (mp.erfc(-bb/mp.sqrt(2))-mp.erfc(-aa/mp.sqrt(2)))/2
        value,bound,use=normal_interval(np.array([a]),np.array([b]))
        rel=float(abs(mp.mpf(float(value[0]))-truth)/truth) if truth else None
        rows.append(dict(lo=a,hi=b,value=float(value[0]),reference=str(truth),relative_error=rel,midpoint=bool(use[0]),quadrature_bound=float(bound[0])))
        if abs(a)<21 and b-a>1e-320:assert rel<1e-12
    out=dict(cases=rows,cpu_seconds=time.process_time()-start,complete=True,underflow_case_excluded_from_relative_assertion=True)
    Path(__file__).with_name('tail_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(cpu_seconds=out['cpu_seconds'],max_relative_nonsubnormal=max(r['relative_error'] for r in rows[:-1]))))
if __name__=='__main__':main()
