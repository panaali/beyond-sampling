"""Frozen positive mixed-measure closure checks through three fresh inputs."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from mixed_volume import Operator,PHI
from two_input import Law
from third_step import push

HERE=Path(__file__).resolve().parent


def main():
    started=time.process_time();deadline=started+110
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    result={'classification':'positive finite-volume mixed-measure approximation; no discretization certificate',
        'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_cutoff':6.,'source_order':2,
        'rows':[],'density_checks':[],'random_model_inputs':0}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        # Positive line-source Jacobian must reproduce the known step2 area law.
        op=Operator(model,8);first,_=op.step(op.initial());law=Law(model)
        points=[np.array([.2,.2]),np.array([.2,1.]),np.array([1.,2.])]
        for s,x in ((.2,.4),(.5,1.),(1.,2.)):
            points.append(np.maximum(model[1]@(model[0]*s)+model[0]*x,0))
        errors=[]
        for h in points:
            if np.any(h<=0):continue
            expected=sum(law.area_density(c,h) for c in law.cells if c['rank']==2)
            actual=op.area_density(first,h);errors.append(abs(expected-actual));assert abs(expected-actual)<1e-12
        result['density_checks'].append({'model':name,'point_checks':len(errors),'max_error':max(errors),'work':dict(op.work)})
        if name=='toy':
            uniform=op.empty(np.array([1.,2.]));uniform.area=np.diff(uniform.edges[0])[:,None]*np.diff(uniform.edges[1])[None,:]/2
            checks=[];evals=0
            for y in ([.5,1.],[1.,1.5],[3.,4.]):
                y=np.array(y);a=op.inverse@y;b=op.inverse@op.wi
                points=sorted(set(float((a[i]-edge)/b[i]) for i in (0,1) for edge in (0.,uniform.box[i]) if -6<(a[i]-edge)/b[i]<6))
                def integrand(x):
                    h=op.inverse@(y-op.wi*x)
                    return float(PHI(x))/(2*op.det) if np.all(h>=0) and np.all(h<=uniform.box) else 0.
                expected,error,info=quad(integrand,-6,6,points=points,epsabs=1e-12,full_output=True)
                actual=op.area_density(uniform,y);checks.append(abs(expected-actual));evals+=info['neval']
                assert abs(expected-actual)<1e-12
            result['uniform_area_convolution_check']={'points':len(checks),'max_error':max(checks),'scalar_integrand_evaluations':evals,'work':dict(op.work)}
        references={2:Law(model).summary(),3:push(model)}
        for n in (8,16,32):
            clock=time.process_time();op=Operator(model,n);state=op.initial();rows=[]
            for t in (1,2,3):
                if time.process_time()>deadline:raise TimeoutError('cycle deadline')
                previous_origin=state.origin;state,flux=op.step(state);hist=op.histogram_output(state)
                masses=state.masses();rayerror=abs(masses['input_ray']-previous_origin*(ndtr(6)-.5))
                assert rayerror<1e-14
                assert abs(hist.sum()-sum(masses.values()))<1e-11
                row={'step':t,'stratum_masses':masses,'integrated_output_masses':flux.tolist(),
                    'histogram_output_masses':hist.tolist(),'ray_regeneration_error':rayerror,
                    'retained_mass':sum(masses.values()),'known_input_tail_omission':1-op.retained**t,
                    'mass_error_from_truncated_law':sum(masses.values())-op.retained**t,
                    'integrated_vs_histogram_output_gap':float(np.max(abs(flux-hist)))}
                if t in references:
                    ref=references[t];row['reference']=ref
                    row['max_stratum_error']=max(abs(v-ref['stratum_masses'][k]) for k,v in masses.items())
                    row['max_integrated_output_error']=float(np.max(abs(flux-ref['output_masses'])))
                    row['max_histogram_output_error']=float(np.max(abs(hist-ref['output_masses'])))
                rows.append(row)
            result['rows'].append({'model':name,'grid':n,'steps':rows,'work':dict(op.work),'cpu_seconds':time.process_time()-clock})
            (HERE/'mixed_volume_results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'model':name,'grid':n,'step3_stratum_error':rows[-1]['max_stratum_error'],
                'step3_output_error':rows[-1]['max_integrated_output_error'],'cpu':time.process_time()-clock}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    (HERE/'mixed_volume_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
