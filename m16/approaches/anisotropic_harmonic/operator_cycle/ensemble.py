"""Fixed prior17 cases, cold dense versus equivalent real operator."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from adapter import estimate,dense_estimate

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons/m16/approaches')
weights=json.loads((ROOT/'small_h2_34/weights_seed0.json').read_text())['T3']
h2=tuple(np.array(weights[k]) for k in ('Wi','Wh','Wo'))
truth2=json.loads((ROOT/'small_h2_34/results_t3.json').read_text())['rows']
rng=np.random.default_rng(47001)
cases=[('H2',0,h2,truth2[0]['accuracy'])]
for i in range(8):
    cases.append(('H2',i+1,tuple(w+.05*rng.normal(size=w.shape) for w in h2),truth2[i+1]['accuracy']))
z=np.load(ROOT/'arc_m43_repro/m43_weights.npz')
h4=tuple(z[k].astype(float) for k in ('Wi','Wh','Wo'))
truth4=json.loads((ROOT/'arc_m43_repro/results.json').read_text())['seeds']['0']['rows']
rng=np.random.default_rng(0)
for i in range(8):
    cases.append(('H4',i,tuple(w+.05*rng.standard_normal(w.shape) for w in h4),truth4[i]['truth']))
start=time.process_time();out=dict(rows=[],random_model_inputs=0,
    frozen_rule='N8 M4 Q64, identical17 prior cases, no degree tuning',
    source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest()
                   for f in ('operator_space.py','adapter.py')})
for family,index,model,truth in cases:
    row=dict(family=family,copy=index,reference=truth,
        model_sha256=hashlib.sha256(b''.join(a.astype('<f8').tobytes() for a in model)).hexdigest())
    try:
        for kind,function in (('dense',dense_estimate),('real_operator',estimate)):
            kwargs=dict(seconds=100-(time.process_time()-start))
            if kind=='real_operator':kwargs['vectorized']=True
            value,stats=function(model,**kwargs)
            row[kind]=dict(estimate=value,error=value-truth,stats=stats)
        row['parity_error']=abs(row['dense']['estimate']-row['real_operator']['estimate'])
        if row['parity_error']>1e-9:raise ArithmeticError('fixed-rule parity failed')
    except Exception as exc:row['failure']=repr(exc)
    out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
    (HERE/'ensemble.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(family=family,copy=index,parity=row.get('parity_error'),
                         failure=row.get('failure'))),flush=True)
    if 'failure' in row:raise SystemExit(1)
out['aggregate']={}
for family in ('H2','H4'):
    rows=[r for r in out['rows'] if r['family']==family]
    out['aggregate'][family]={method:dict(
        rmse=float(np.sqrt(np.mean([r[method]['error']**2 for r in rows]))),
        mean_cpu=float(np.mean([r[method]['stats']['cpu_seconds'] for r in rows])),
        mean_accounted_arithmetic=float(np.mean([r[method]['stats']['accounted_scalar_arithmetic_upper'] for r in rows])),
        mean_squared_error_times_cpu=float(np.mean([r[method]['error']**2*r[method]['stats']['cpu_seconds'] for r in rows])))
        for method in ('dense','real_operator')}
    out['aggregate'][family]['maximum_parity_error']=max(r['parity_error'] for r in rows)
out['complete']=True
(HERE/'ensemble.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['aggregate']),flush=True)
