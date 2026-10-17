import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import betainc,ndtr
from driver import rule,setup,estimate
HERE=Path(__file__).resolve().parent
start=time.process_time();out={'rule_rows':[],'full_rows':[]}
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=a[0].ravel(),a[1],a[2]
data=setup(model,9,15.)
alpha=ndtr(-6.);z=1-2*alpha;qm,qp=(ndtr(np.array([-4.,4.]))-alpha)/z
for name,points in (('artificial',np.linspace(-4.,4.,65)),('trained',data['points'])):
    for coordinate in ('q','u'):
        for order in (2,4):
            q,weights,stats=rule(points,4,order,coordinate)
            errors=[]
            for power in range(3):
                exact=12/((3+power)*(4+power))*(betainc(3+power,2,qp)-betainc(3+power,2,qm))
                errors.append(float(weights@q**power-exact))
            out['rule_rows'].append(dict(name=name,coordinate=coordinate,order=order,moment_errors=errors,stats=stats))
            if order==4:assert max(map(abs,errors))<2e-9
for name,wo in (('constant',np.zeros((4,2))),('duplicate',np.tile(model[2][0],(4,1)))):
    for coordinate in ('q','u'):
        value,stats=estimate((model[0],model[1],wo),3,4,coordinate,seconds=15.-(time.process_time()-start))
        expected=stats['raw_target_mass']/4
        out['full_rows'].append(dict(name=name,coordinate=coordinate,estimate=value,expected_from_raw_mass=expected,error=value-expected,stats=stats))
        assert abs(value-expected)<1e-10
out['total_cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=out['total_cpu_seconds'],max_full_error=max(abs(r['error']) for r in out['full_rows']),trained_panels=len(data['points'])-1)))
