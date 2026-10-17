"""Fourth-step mixed-measure comparison with independent R4 cone geometry."""
from collections import defaultdict
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from mixed_volume import Operator

HERE=Path(__file__).resolve().parent
GEOMETRY=Path('/workspace/clean-neurons/m16/approaches/small_h2_56/geometry.py')
VOLUME=Path('/workspace/clean-neurons/m16/approaches/small_h4_4/stable_volume.py')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def reference(model,deadline):
    started=time.process_time();geo=module('strata_four_geometry',GEOMETRY);vol=module('strata_four_volume',VOLUME)
    g=geo.Geometry(seconds=30);wi,wh,wo=model
    cells=g.gates((wi,wh,np.zeros((4,2))))
    strata={k:[] for k in ('origin','axis0','axis1','input_ray','area')};outputs=[[] for _ in wo]
    for vertices,B in cells:
        if not np.any(B):kind='origin'
        elif not np.any(B[1]):kind='axis0'
        elif not np.any(B[0]):kind='axis1'
        elif np.linalg.matrix_rank(B,tol=1e-12*np.linalg.norm(B))==1:kind='input_ray'
        else:kind='area'
        strata[kind].append(vertices);lines=wo@B
        for label in range(len(wo)):
            poly=vertices
            for other in range(len(wo)):
                if label==other:continue
                normal=lines[label]-lines[other]
                if not np.any(normal) and other<label:poly=None;break
                poly=g.clip(poly,normal)
                if poly is None:break
            if poly is not None:outputs[label].append(poly)
    triangulated={k:g.simplices(pieces) for k,pieces in strata.items()}
    output_rays=[g.simplices(pieces) for pieces in outputs]
    construction_cpu=time.process_time()-started;work=defaultdict(float);cache={}
    def integrate(rays):
        value=bound=0.
        for ray in rays:
            if time.process_time()>deadline:raise TimeoutError('four-dimensional reference deadline')
            key=ray.tobytes();work['simplex_queries']+=1
            if key in cache:p,e=cache[key];work['simplex_cache_hits']+=1
            else:
                p,e=vol.mass(ray,20);cache[key]=(p,e);work['unique_simplex_integrations']+=1
                work['quadrature_nodes']+=20**3
            value+=p;bound+=e
        return value,bound
    masses={};bounds={}
    for kind,rays in triangulated.items():masses[kind],bounds[kind]=integrate(rays)
    probabilities=[];output_bounds=[]
    for rays in output_rays:
        p,e=integrate(rays);probabilities.append(p);output_bounds.append(e)
    return {'stratum_masses':masses,'stratum_integration_bounds':bounds,'output_masses':probabilities,
        'output_integration_bounds':output_bounds,'full_stratum_mass':sum(masses.values()),
        'full_output_mass':sum(probabilities),'geometry_counts':g.counts,'integration_work':dict(work),
        'geometry_cpu_seconds':construction_cpu,'total_cpu_seconds':time.process_time()-started,
        'classification':'integration-only bounds conditional on floating cone geometry and arithmetic'}


def main():
    started=time.process_time();deadline=started+110
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (GEOMETRY,VOLUME,HERE/'mixed_volume.py')}
    result={'classification':'four-input marginal validation, not T8 rank accuracy','sources':sources(),
        'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':[],'random_model_inputs':0}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        ref=reference(model,deadline)
        assert abs(ref['full_stratum_mass']-1)<=sum(ref['stratum_integration_bounds'].values())+1e-10
        assert abs(ref['full_output_mass']-1)<=sum(ref['output_integration_bounds'])+1e-10
        for n in (16,32):
            if time.process_time()>deadline:raise TimeoutError('mixed transport deadline')
            clock=time.process_time();op=Operator(model,n);state=op.initial()
            for _ in range(4):state,output=op.step(state)
            histogram=op.histogram_output(state);mass=state.masses()
            row={'model':name,'grid':n,'reference':ref,'stratum_masses':mass,
                'integrated_output_masses':output.tolist(),'histogram_output_masses':histogram.tolist(),
                'max_stratum_error':max(abs(v-ref['stratum_masses'][k]) for k,v in mass.items()),
                'max_integrated_output_error':float(np.max(abs(output-ref['output_masses']))),
                'max_histogram_output_error':float(np.max(abs(histogram-ref['output_masses']))),
                'known_input_tail_omission':1-op.retained**4,'mass_error_from_truncated_law':sum(mass.values())-op.retained**4,
                'transport_work':dict(op.work),'transport_cpu_seconds':time.process_time()-clock}
            result['rows'].append(row);(HERE/'fourth_step_results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'model':name,'grid':n,'stratum_error':row['max_stratum_error'],
                'output_error':row['max_integrated_output_error'],'transport_cpu':row['transport_cpu_seconds'],
                'reference_cpu':ref['total_cpu_seconds']}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();result['source_files_changed_during_run']=result['sources']!=result['sources_after']
    (HERE/'fourth_step_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
