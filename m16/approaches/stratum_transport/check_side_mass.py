"""Exact-form third-step side mass reference and saved-operator comparison."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from third_step import push

HERE=Path(__file__).resolve().parent

def main():
    started=time.process_time();deadline=started+20
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    old=json.loads((HERE/'interface_results.json').read_text())
    paths=[HERE/p for p in ('check_side_mass.py','third_step.py','two_input.py','interface_results.json')]
    result={'classification':'exact-form floating three-input side-mass reference','sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'random_model_inputs':0,'rows':[]}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        w,W,_=model;n=np.array([-w[1],w[0]])
        clock=time.process_time();ref=push((w,W,np.array([n,-n])),deadline)
        masses=ref['stratum_masses'];output=ref['output_masses']
        side=[output[0]-masses['origin']-masses['input_ray']-masses['axis1'],output[1]-masses['axis0']]
        assert abs(sum(side)-masses['area'])<1e-12 and min(side)>-1e-12
        rows=[]
        for previous in old['rows']:
            if previous['model']==name:
                actual=previous['steps'][2]['side_masses']
                rows.append({'grid':previous['grid'],'reconstruction':previous['reconstruction'],'side_masses':actual,'signed_side_errors':(np.array(actual)-side).tolist()})
        row={'model':name,'reference_side_masses':side,'reference':ref,'comparisons':rows,'reference_cpu_seconds':time.process_time()-clock}
        result['rows'].append(row);print(json.dumps(row),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    (HERE/'side_mass_results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
