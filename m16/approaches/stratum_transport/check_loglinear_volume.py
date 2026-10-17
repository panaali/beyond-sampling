"""Fixed log-linear comparison and regression controls for T83."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from affine_volume import AffineOperator
from loglinear_volume import LoglinearOperator
from check_affine_volume import synthetic,axis_control

HERE=Path(__file__).resolve().parent

def constant_control(model):
    op=LoglinearOperator(model,6);op.pending_kind='upper';state=op.empty(np.ones(2))
    for i in range(op.n):
        for j in range(op.n):state.area[i,j]=op.area_cell_volume(state,i,j)
    H,P=op.source_nodes(state)
    assert np.max(abs(op.area_gradients))<1e-12
    assert abs(P.sum()-.5)<1e-13 and np.all(P>0)
    assert np.max(abs(P@H-[1/6,1/3]))<1e-13
    return {'constant_gradient_max':float(np.max(abs(op.area_gradients))),'work':dict(op.work)}

def main():
    started=time.process_time();deadline=started+110
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    refs3=json.loads((HERE/'third_step_results.json').read_text())['rows']
    refs4=json.loads((HERE/'fourth_step_results.json').read_text())['rows']
    old=json.loads((HERE/'affine_results.json').read_text())['rows']
    paths=[HERE/s for s in ('mixed_volume.py','support_volume.py','quadrature_volume.py','affine_volume.py',
        'loglinear_volume.py','check_loglinear_volume.py','affine_results.json','third_step_results.json','fourth_step_results.json')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'classification':'local log-linear reconstruction with positive cell normalization','sources':sources(),
        'random_model_inputs':0,'affine_synthetic_regressions':[synthetic(toy,k,g) for k in ('full','upper') for g in ((.2,.3),(.2,-.3))],
        'axis_regression':axis_control(toy),'constant_control':constant_control(toy),'rows':[],'coordinate_controls':[]}
    def run(model,n,cls=LoglinearOperator):
        clock=time.process_time();op=cls(model,n);state=op.initial();steps=[]
        for t in range(1,5):
            if time.process_time()>deadline:raise TimeoutError('log-linear diagnostic CPU deadline')
            state,output=op.step(state);mass=state.masses()
            assert abs(sum(mass.values())-op.retained**t)<1e-12
            steps.append({'stratum_masses':mass,'output_masses':output.tolist()})
        return {'grid':n,'steps':steps,'work':dict(op.work),'cpu_seconds':time.process_time()-clock}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        # A full n16 affine trajectory verifies unchanged shared hooks.
        regression=run(model,16,AffineOperator);prior=next(r for r in old if r['model']==name and r['grid']==16)
        difference=max(abs(regression['steps'][t]['stratum_masses'][k]-prior['steps'][t]['stratum_masses'][k])
            for t in range(4) for k in prior['steps'][t]['stratum_masses'])
        assert difference<1e-12
        result.setdefault('affine_trajectory_regressions',[]).append({'model':name,'max_difference':difference,'run':regression})
        for n in (16,32):
            row=run(model,n);row['model']=name
            third=next(r['forward'] for r in refs3 if r['model']==('toy' if name=='toy' else 'trained_H2T8_three_steps'))
            fourth=next(r['reference'] for r in refs4 if r['model']==name and r['grid']==n)
            row['errors']=[]
            for t,ref in ((3,third),(4,fourth)):
                errors={k:v-ref['stratum_masses'][k] for k,v in row['steps'][t-1]['stratum_masses'].items()}
                row['errors'].append({'step':t,'signed_stratum_errors':errors,'max_stratum_error':max(abs(v) for v in errors.values()),
                    'max_output_error':float(np.max(abs(np.array(row['steps'][t-1]['output_masses'])-ref['output_masses'])))})
            row['old_affine']=next(r for r in old if r['model']==name and r['grid']==n)
            result['rows'].append(row)
            print(json.dumps({'model':name,'grid':n,'errors':row['errors'],'cpu':row['cpu_seconds']}),flush=True)
        D=np.diag([2.,.5]);inverse=np.linalg.inv(D);changed=run((D@model[0],D@model[1]@inverse,model[2]@inverse),16)
        base=next(r for r in result['rows'] if r['model']==name and r['grid']==16)
        difference=max(abs(changed['steps'][t]['stratum_masses'][k]-base['steps'][t]['stratum_masses'][k]) for t in range(4) for k in base['steps'][t]['stratum_masses'])
        difference=max(difference,max(float(np.max(abs(np.array(changed['steps'][t]['output_masses'])-base['steps'][t]['output_masses']))) for t in range(4)))
        assert difference<1e-12
        changed.update(model=name,max_coordinate_discrepancy=difference);result['coordinate_controls'].append(changed)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources_after']==result['sources']
    (HERE/'loglinear_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
