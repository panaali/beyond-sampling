"""Existing-panel sensitivity to a fixed cold geometry cost; no new models."""
import json,time,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
start=time.process_time();d=json.loads((HERE/'absolute_k24_s4.json').read_text());rows=d['rows'];n=len(rows)
p=np.array([r['truth'] for r in rows]);pv=p*(1-p);e2=np.array([(r['t75']['value']-r['truth'])**2 for r in rows]);ops=np.array([r['t75']['own_ops'] for r in rows]);sample=np.array([r['t75']['sample_ops'] for r in rows]);protocol='--protocol' in sys.argv
if protocol:sample=sample-3
rng=np.random.default_rng(991);ix=rng.integers(n,size=(1000,n));bm=np.mean(e2[ix],axis=1)
def calc(c):
 var=pv*sample/(ops+c);point=float(np.mean(e2)/np.mean(var));bs=bm/np.mean(var[ix],axis=1)
 return dict(extra_scalar_ops=c,ratio=point,bootstrap_ci=np.quantile(bs,[.025,.975]).tolist())
def crossing(upper):
 lo=0.;hi=1.
 def v(c):
  z=calc(c);return z['bootstrap_ci'][1] if upper else z['ratio']
 while v(hi)<1:hi*=2
 for _ in range(40):
  mid=(lo+hi)/2
  if v(mid)<1:lo=mid
  else:hi=mid
 return dict(last_below=lo,first_at_or_above=hi,at_crossing=calc(hi))
out=dict(panel='absolute_k24_s4',n=n,cost_convention='same scalar arithmetic convention as own T75;115/sample historically,112/sample with --protocol label exemption; c is added once to every model, independently of its outcome',grid=[calc(c) for c in(0,512,1024,2048,4096,8192,16384)],point_crossing=crossing(False),upper_ci_crossing=crossing(True),cpu=time.process_time()-start,scope='Threshold sensitivity only. No claim that actual cold setup falls below these thresholds; model-independent numpy geometry was not fully instrumented. Reusable universal constants are different from model-specific memoization; no model-dependent values are reused between estimates.')
(HERE/('setup_sensitivity_protocol.json' if protocol else 'setup_sensitivity.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
