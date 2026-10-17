"""Fresh arc-rule check against already independently saved MP integrals.

Check normalized absolute remainder arithmetic and orientation; numerical
rounding is explicitly outside the analytic GL remainder.
"""
import json,math,sys,time
from pathlib import Path
import mpmath as mp
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h2_78/m16/approaches/real_arc_tail')
sys.path.insert(0,str(SOURCE))
from quadrature import ArcRule
mp.mp.dps=80;start=time.process_time();rule=ArcRule();rows=[]
saved=json.loads((HERE/'positive_arc_review_results.json').read_text())
a,b=.37,-.81
for theta in (.73,2.4):
    values,bounds=rule.evaluate(a,b,theta,6)
    rev,revbounds=rule.evaluate(a*math.cos(theta)+b*math.sin(theta),a*math.sin(theta)-b*math.cos(theta),theta,6)
    C=mp.factorial(8)**4/(17*mp.factorial(16)**3)
    M=max(1,math.ceil(float(mp.mpf(theta)*6*(C/mp.mpf('1e-13'))**(mp.mpf(1)/16))))
    for row in saved['rows']:
        if row['theta']!=theta or row['degree']>6:continue
        n=row['degree'];i=n//2;reference=mp.mpf(row['reference'])
        relative=float(abs(mp.mpf(values[i])-reference)/reference)
        assert relative<3e-13 and abs(values[i]-rev[i])<3e-13*float(reference)
        bound_ref=mp.mpf(0) if n==0 else C*mp.mpf(theta)*(mp.mpf(theta)*n/M)**16*(mp.mpf(a)**2+mp.mpf(b)**2)**(mp.mpf(n)/2)
        if n:assert abs(mp.mpf(bounds[i])-bound_ref)/bound_ref<mp.mpf('2e-14')
        rows.append(dict(theta=theta,degree=n,panels=M,relative_error=relative,integration_bound=bounds[i],independent_bound=mp.nstr(bound_ref,50),observed_error=float(abs(mp.mpf(values[i])-reference))))
out=dict(rows=rows,cpu_seconds=time.process_time()-start,max_relative_error=max(r['relative_error'] for r in rows),
         scope='Independent exact remainder formula and saved MP reference; no roundoff enclosure or conditional-query rerun.')
(HERE/'arc_rule_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
