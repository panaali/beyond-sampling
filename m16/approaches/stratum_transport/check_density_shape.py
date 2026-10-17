"""Frozen single-stage density-shape falsifier with preserved cell masses."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from density_shape import ShapeOperator
from mixed_volume import PHI

HERE=Path(__file__).resolve().parent

def main():
    started=time.process_time();deadline=started+20
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    reference=json.loads((HERE/'third_step_results.json').read_text())['rows']
    baseline=json.loads((HERE/'quadrature_results.json').read_text())['rows']
    paths=[HERE/s for s in ('mixed_volume.py','support_volume.py','quadrature_volume.py','density_shape.py',
        'check_density_shape.py','two_input.py','third_step_results.json','quadrature_results.json')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'classification':'one-stage two-input density oracle; not a scalable solver','sources':sources(),
        'random_model_inputs':0,'rows':[],'kernel_controls':[]}
    control=ShapeOperator(toy,1)
    for z in (.1,.5,2.,5.,10.):
        lo=max(0.,z,(z-6)/2);hi=min(6.,(z+6)/2)
        expected=quad(lambda s:float(PHI(s)*PHI(z-2*s)),lo,hi,epsabs=1e-13)[0] if hi>lo else 0.
        actual=float(control.density('axis1',np.array([[0.,z]]))[0]);assert abs(actual-expected)<2e-14
        result['kernel_controls'].append({'axis_coordinate':z,'density':actual,'independent_density':expected})
    for h in ((.1,.2),(.3,.8),(1.,1.5),(2.,3.),(.1,7.),(8.,9.)):
        s=h[1]-h[0];x=2*h[0]-h[1]
        expected=float(PHI(s)*PHI(x)) if 0<s<=6 and abs(x)<=6 else 0.
        actual=float(control.density('area',np.array([h]))[0]);assert abs(actual-expected)<2e-14
        result['kernel_controls'].append({'hidden':list(h),'density':actual,'independent_density':expected})
    result['kernel_control_work']=dict(control.work)
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        ref=next(r['forward'] for r in reference if r['model']==('toy' if name=='toy' else 'trained_H2T8_three_steps'))
        for n in (16,32):
            for shape in ('uniform','area','axis','both'):
                clock=time.process_time();op=ShapeOperator(model,n,shape);state=op.initial()
                for _ in range(3):
                    if time.process_time()>deadline:raise TimeoutError('density shape diagnostic deadline')
                    state,output=op.step(state)
                mass=state.masses();errors={k:v-ref['stratum_masses'][k] for k,v in mass.items()}
                if shape=='uniform':
                    old=next(r for r in baseline if r['model']==name and r['grid']==n and r['source_order']==4)
                    assert max(abs(v-old['steps'][2][k]) for k,v in mass.items())<1e-12
                assert abs(sum(mass.values())-op.retained**3)<1e-12
                row={'model':name,'grid':n,'shape':shape,'stratum_masses':mass,'signed_stratum_errors':errors,
                    'max_stratum_error':max(abs(v) for v in errors.values()),'output_masses':output.tolist(),
                    'max_output_error':float(np.max(abs(output-ref['output_masses']))),'work':dict(op.work),
                    'exact_geometry_work':dict(op.law.work),'cpu_seconds':time.process_time()-clock}
                result['rows'].append(row)
                print(json.dumps({k:row[k] for k in ('model','grid','shape','max_stratum_error','max_output_error','cpu_seconds')}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources']==result['sources_after']
    (HERE/'density_shape_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
