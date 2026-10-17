"""Frozen model/noise/degeneracy checks; no new input samples."""
import json,time
import numpy as np
from check_two_input import HERE,Law,angular_reference
from check_third_step import spherical_reference
from third_step import push

start=time.process_time();deadline=start+20
import torch
state=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth',map_location='cpu',weights_only=True)
base=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
base=(base[0].reshape(-1),base[1],base[2]);models=[]
for sigma in (.0025,.02):
    rng=np.random.default_rng(1)
    models.append((f'public_T8_noise_{sigma}',tuple(W+sigma*np.sqrt(np.mean(W**2))*rng.standard_normal(W.shape) for W in base)))
data=np.load(HERE.parent/'small_h2_78'/'LOCALPILOT_H2_T7_seed0.npz')
models.append(('LOCALPILOT_T7',tuple(data[k].astype(float) for k in ('Wi','Wh','Wo'))))
wo=np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]])
models.extend([('zero_input',(np.zeros(2),np.eye(2),wo)),('collinear',(np.ones(2),np.eye(2),wo)),
    ('opposite_input',(np.array([1.,-1.]),np.array([[.25,.5],[.5,.25]]),wo))])
result={'rows':[],'random_model_inputs':0,'classification':'fixed short-horizon primitive checks'}
for name,model in models:
    clock=time.process_time();law=Law(model);two=law.summary();mass,output,work=angular_reference(model)
    two_error=max(max(abs(two['stratum_masses'][k]-v) for k,v in mass.items()),float(np.max(abs(np.array(two['output_masses'])-output))))
    three=push(model,deadline);ref=spherical_reference(model)
    three_error=max(max(abs(three['stratum_masses'][k]-v) for k,v in ref['stratum_masses'].items()),
        float(np.max(abs(np.array(three['output_masses'])-ref['output_masses']))))
    assert max(two_error,three_error)<1e-10,(name,two,three,ref)
    for row in (two,three):
        assert abs(sum(row['stratum_masses'].values())-1)<1e-10
        assert abs(sum(row['output_masses'])-1)<1e-10
        if model[0][0]*model[0][1]<0:assert abs(row['stratum_masses']['input_ray'])<1e-12
        if name=='zero_input':assert abs(row['stratum_masses']['origin']-1)<1e-12
        if name=='collinear':assert abs(row['stratum_masses']['origin']+row['stratum_masses']['input_ray']-1)<1e-12
    result['rows'].append({'model':name,'two_step':two,'three_step':three,'reference_three_step':ref,
        'two_step_reference_work':work,'two_step_error':two_error,'three_step_error':three_error,
        'cpu_seconds':time.process_time()-clock})
    (HERE/'stress_results.json').write_text(json.dumps(result,indent=2)+'\n')
result['cpu_seconds_including_torch_import']=time.process_time()-start
(HERE/'stress_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cpu':result['cpu_seconds_including_torch_import'],'rows':len(result['rows']),
    'max_error':max(max(r['two_step_error'],r['three_step_error']) for r in result['rows'])}))
