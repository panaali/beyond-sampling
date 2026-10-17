"""Fresh-weight checks of frozen split-log/event recipe against uniform."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from quadrature_volume import QuadratureOperator
from event_volume import EventInterfaceOperator
from check_fourth_step import reference,GEOMETRY,VOLUME

HERE=Path(__file__).resolve().parent

def main():
    started=time.process_time();deadline=started+110
    import torch
    checkpoint=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    base=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    base=(base[0].reshape(-1),base[1],base[2])
    paths=[HERE/p for p in ('mixed_volume.py','support_volume.py','quadrature_volume.py','affine_volume.py',
        'loglinear_volume.py','interface_volume.py','event_volume.py','check_interface_noise.py','check_fourth_step.py')]+[GEOMETRY,VOLUME]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'classification':'fresh-weight four-input marginal validation, not rank accuracy','sources':sources(),
        'random_model_inputs':0,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'rows':[]}
    for sigma,seed in ((.0025,20261017),(.02,20261018)):
        rng=np.random.default_rng(seed)
        model=tuple(W+sigma*np.sqrt(np.mean(W**2))*rng.standard_normal(W.shape) for W in base)
        ref=reference(model,deadline)
        assert abs(ref['full_stratum_mass']-1)<sum(ref['stratum_integration_bounds'].values())+1e-10
        assert abs(ref['full_output_mass']-1)<sum(ref['output_integration_bounds'])+1e-10
        rows=[]
        for variant in ('uniform','split_log_event'):
            clock=time.process_time()
            op=QuadratureOperator(model,32,4) if variant=='uniform' else EventInterfaceOperator(model,32,'loglinear',True)
            state=op.initial()
            for _ in range(4):
                if time.process_time()>deadline:raise TimeoutError('fresh-weight CPU deadline')
                state,output=op.step(state)
            mass=state.masses();assert abs(sum(mass.values())-op.retained**4)<1e-12
            rows.append({'variant':variant,'grid':32,'stratum_masses':mass,'output_masses':output.tolist(),
                'signed_stratum_errors':{k:v-ref['stratum_masses'][k] for k,v in mass.items()},
                'max_stratum_error':max(abs(v-ref['stratum_masses'][k]) for k,v in mass.items()),
                'max_output_error':float(np.max(abs(output-ref['output_masses']))),
                'known_input_tail_omission':1-op.retained**4,'work':dict(op.work) if variant=='uniform' else op.report(),
                'cpu_seconds':time.process_time()-clock})
        row={'sigma':sigma,'seed':seed,'weights':[W.tolist() for W in model],'reference':ref,'variants':rows}
        result['rows'].append(row)
        print(json.dumps({'sigma':sigma,'seed':seed,'variants':[{k:r[k] for k in ('variant','max_stratum_error','max_output_error','cpu_seconds')} for r in rows],
            'reference_cpu':ref['total_cpu_seconds']}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources']==result['sources_after']
    (HERE/'interface_noise_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
