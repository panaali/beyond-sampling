"""Log-linear reconstruction on separate sides, fixed event-aligned rule."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from event_volume import EventInterfaceOperator
from check_interface_volume import synthetic,thin_control

HERE=Path(__file__).resolve().parent

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
    old=json.loads((HERE/'loglinear_results.json').read_text())['rows']
    sides=json.loads((HERE/'side_mass_results.json').read_text())['rows']
    paths=[HERE/p for p in ('mixed_volume.py','support_volume.py','quadrature_volume.py','affine_volume.py',
        'loglinear_volume.py','interface_volume.py','event_volume.py','check_split_loglinear.py',
        'loglinear_results.json','side_mass_results.json')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'classification':'same-side log-linear with event-aligned positive quadrature','sources':sources(),
        'random_model_inputs':0,'synthetic_control':synthetic(toy,'loglinear'),
        'thin_zero_cell_control':thin_control('loglinear'),'rows':[],'coordinate_controls':[]}
    def run(model,n):
        clock=time.process_time();op=EventInterfaceOperator(model,n,'loglinear',True);state=op.initial();steps=[]
        for t in range(1,5):
            if time.process_time()>deadline:raise TimeoutError('split log-linear CPU deadline')
            state,output=op.step(state);mass=state.masses()
            assert abs(sum(mass.values())-op.retained**t)<1e-12
            steps.append({'stratum_masses':mass,'output_masses':output.tolist(),'side_masses':[float(s.area.sum()) for s in state.side_states]})
        return {'grid':n,'steps':steps,'work':op.report(),'cpu_seconds':time.process_time()-clock}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        for n in (16,32):
            row=run(model,n);row['model']=name;row['errors']=[]
            third=next(r['forward'] for r in refs3 if r['model']==('toy' if name=='toy' else 'trained_H2T8_three_steps'))
            fourth=next(r['reference'] for r in refs4 if r['model']==name and r['grid']==n)
            for t,ref in ((3,third),(4,fourth)):
                error={k:v-ref['stratum_masses'][k] for k,v in row['steps'][t-1]['stratum_masses'].items()}
                row['errors'].append({'step':t,'signed_stratum_errors':error,'max_stratum_error':max(abs(v) for v in error.values()),
                    'max_output_error':float(np.max(abs(np.array(row['steps'][t-1]['output_masses'])-ref['output_masses'])))})
            row['signed_step3_side_errors']=(np.array(row['steps'][2]['side_masses'])-next(r['reference_side_masses'] for r in sides if r['model']==name)).tolist()
            row['prior_unsplit']=next(r for r in old if r['model']==name and r['grid']==n)
            result['rows'].append(row)
            print(json.dumps({'model':name,'grid':n,'errors':row['errors'],'cpu':row['cpu_seconds']}),flush=True)
        D=np.diag([2.,.5]);inv=np.linalg.inv(D);changed=run((D@model[0],D@model[1]@inv,model[2]@inv),16)
        base=next(r for r in result['rows'] if r['model']==name and r['grid']==16)
        difference=max(abs(changed['steps'][t]['stratum_masses'][k]-base['steps'][t]['stratum_masses'][k]) for t in range(4) for k in base['steps'][t]['stratum_masses'])
        difference=max(difference,max(float(np.max(abs(np.array(changed['steps'][t]['output_masses'])-base['steps'][t]['output_masses']))) for t in range(4)))
        assert difference<1e-12
        changed.update(model=name,max_coordinate_discrepancy=difference);result['coordinate_controls'].append(changed)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources_after']==result['sources']
    (HERE/'split_loglinear_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
