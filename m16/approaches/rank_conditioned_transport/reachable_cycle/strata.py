"""Same frozen nodes, with explicit state strata and terminal zero-tie mass."""
import hashlib
import json
from pathlib import Path
import torch
from diagnostic import evaluate

HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in (
    'rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
result=evaluate(model,16,8,strata=True)
result['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
result['diagnostic_source_sha256']=hashlib.sha256((HERE/'diagnostic.py').read_bytes()).hexdigest()
prior=json.loads((HERE/'pilot.json').read_text())['rows'][1]
assert abs(prior['source_quadrature_estimate']-result['source_quadrature_estimate'])<1e-15
for before,after in zip(prior['rows'],result['rows']):
    assert before['mode']==after['mode'] and before['hidden_order']==after['hidden_order']
    for key in ('signed','absolute','squared'):assert abs(before[key]-after[key])<1e-15
(HERE/'strata.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('stratum_mass','stratum_zero_tie_mass',
    'no_other_positive_volume_ties_by_distinct_affine_slopes','cpu_seconds')}))
for row in result['rows']:
    print(json.dumps({k:row[k] for k in ('hidden_order','mode','strata')}))
