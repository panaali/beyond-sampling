"""Saved-result audits independent of estimator, LP and geometry engines."""
from decimal import Decimal as D,getcontext
from pathlib import Path
import json,time
import numpy as np
getcontext().prec=60
start=time.process_time()
p=Path('/workspace/work/recover-small_h2_56/m16/approaches/training_axis_review/absolute_k24_s4.json')
d=json.loads(p.read_text(),parse_float=D);rows=d['rows'];n=D(len(rows))
loss=[(r['t75']['value']-r['truth'])**2 for r in rows]
var=[r['truth']*(1-r['truth']) for r in rows]
cost=[D(r['t75']['own_ops'])/D(r['t75']['sample_ops']) for r in rows]
claim=[D(r['t75']['claimed_ops'])/D(72) for r in rows]
assert all(0<r['truth']<1 for r in rows) and min(cost)>0
mc=[v/c for v,c in zip(var,cost)]
ratio=sum(loss)/sum(mc)
uniform=(sum(loss)/n)*(sum(cost)/n)/(sum(var)/n)
claimed_uniform=(sum(loss)/n)*(sum(claim)/n)/(sum(var)/n)
# Same frozen resamples, independently aggregate count histograms rather than
# slicing arrays and invoking the author's summary function.
rng=np.random.default_rng(991);l=np.asarray(list(map(float,loss)));v=np.asarray(list(map(float,mc)))
boots=[]
for _ in range(1000):
    counts=np.bincount(rng.integers(len(rows),size=len(rows)),minlength=len(rows))
    boots.append(float(counts@l/(counts@v)))
training={'models':len(rows),'direct_ratio':str(ratio),'uniform_average_cost_ratio':str(uniform),
          'claimed_counter_uniform_ratio':str(claimed_uniform),'bootstrap_ci':np.quantile(boots,[.025,.975]).tolist(),
          'individual_losses':sum(a>=b for a,b in zip(loss,mc)),
          'max_abs_error':str(max(abs(r['t75']['value']-r['truth']) for r in rows)),
          'scope':'pointwise noisy-copy bootstrap; not simultaneous, training-seed-generalization, or rigorous coverage'}

p=Path('/workspace/work/recover-small_h2_910/m16/approaches/bulk_error_mechanism/attribution.json')
d=json.loads(p.read_text(),parse_float=D)
gate=d['sets']['gate']['rules'][-1]['probability'];good=d['sets']['correct']['rules'][-1]['probability']
errors={k:gate.get(k,D(0))-good.get(k,D(0)) for k in d['groups']}
for k,e in errors.items():assert abs(e-d['error_by_group'][k])<D('1e-15')
total=sum(errors.values())
bulk={'errors':{k:str(v) for k,v in errors.items()},'total_error':str(total),
      'remaining_error_share':str(errors['remaining']/total),'gate_total':str(sum(gate.values())),
      'correct_total':str(sum(good.values())),
      'scope':'full all-off resets/exact contrast-column erasure only; no general output-forgetting exclusion'}
out={'training':training,'bulk':bulk,'cpu_seconds':time.process_time()-start}
Path(__file__).with_name('training_bulk_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
