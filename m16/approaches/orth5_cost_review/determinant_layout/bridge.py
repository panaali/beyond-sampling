"""Saved-only integer contraction; no determinant, mass or model calls."""
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'bridge.json'
assert not OUT.exists(),'Preserve prior contraction'
t0=time.process_time()
pins=json.loads((HERE/'bridge_pins.json').read_text())
for name,sha in pins.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
observed=json.loads((HERE/'observed.json').read_text());baseline=json.loads((HERE/'baseline.json').read_text())
assert observed['complete'] and baseline['complete'] and len(observed['rows'])==12
assert observed['baseline_sha256']==pins['baseline.json']
assert sum(len(r['events']) for r in observed['rows'])==774
pilot=json.loads((HERE/'../pilot.json').read_text());assert pilot['complete']
counts=[]
for row in pilot['rows']:
    if row['branch']!='log':continue
    c={3:0,4:0,5:0}
    for event in row['trace']['kernel_events']:
        if event['op']!='np.linalg.det':continue
        shape=event['inputs'][0]['shape'];assert shape[-1]==shape[-2]
        c[shape[-1]]+=shape[0] if len(shape)==3 else 1
    assert c=={3:3840,4:665,5:1};counts.append(c)
assert len(counts)==3
numeric=3840*100+665*171
logs=3840*3+665*4;exps=3840+665
assert numeric==497715 and logs+exps==18685 and numeric+logs+exps==516400
out={'complete':True,'dependency_pins':pins,'review_premises':['26fd4e11','f53ea74a','e54c58e9'],
     'saved_synthetic_log_cases_checked':len(counts),'per_completed_log_branch':{'det3':3840,'det4':665,'det5_unpriced':1},
     'conditional_selected_numeric_category_upper':numeric,'conditional_log_call_upper':logs,'conditional_exp_call_upper':exps,
     'historical_combined_named_category_upper':numeric+logs+exps,
     'source_exposed_expression_units_kept_separate':32262,'cpu_seconds':time.process_time()-t0,
     'native_mass_model_MC_calls':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'scope':'Conditional inherited category envelopes only, after fixed layout/copy/API tests. Neither full native upper/lower, historical-corpus binding, sampler budget nor speed/accuracy claim.'}
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['complete','conditional_selected_numeric_category_upper','conditional_log_call_upper','conditional_exp_call_upper','cpu_seconds']}))
