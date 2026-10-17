"""Independent spherical partition for the short-horizon forward law."""
from collections import defaultdict
import hashlib,itertools,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from third_step import push,primitive

HERE=Path(__file__).resolve().parent
FUND=Path('/workspace/clean-neurons/m16/approaches/feynman_fundamentals')
sys.path.insert(0,str(FUND))
from tail import clip
from moments import area


def spherical_reference(model):
    wi,wh,wo=model;stats=defaultdict(float)
    regions=[(np.diag(sign),np.zeros((2,3))) for sign in itertools.product((-1.,1.),repeat=3)]
    for t in range(3):
        updated=[]
        for polygon,B in regions:
            Z=wh@B;Z[:,t]+=wi;stats['affine_products']+=1
            partial=[(polygon,np.zeros_like(B))]
            for i in (0,1):
                if not np.any(Z[i]):continue
                split=[]
                for p,C in partial:
                    active=clip(p,Z[i],stats);inactive=clip(p,-Z[i],stats)
                    if len(active):D=C.copy();D[i]=Z[i];split.append((active,D))
                    if len(inactive):split.append((inactive,C.copy()))
                partial=split
            updated.extend(partial)
        regions=updated
    mass={k:0. for k in ('origin','axis0','axis1','input_ray','area')};outputs=np.zeros(len(wo))
    for p,B in regions:
        if not np.any(B):kind='origin'
        elif not np.any(B[1]):kind='axis0'
        elif not np.any(B[0]):kind='axis1'
        elif np.linalg.matrix_rank(B,tol=1e-12*np.linalg.norm(B))==1:kind='input_ray'
        else:kind='area'
        mass[kind]+=area(p)/(4*math.pi);stats['area_calls']+=1
        lines=wo@B
        for i in range(len(wo)):
            polygon=p.copy()
            for j in range(len(wo)):
                if j==i:continue
                margin=lines[i]-lines[j];stats['readout_comparisons']+=1
                if not np.any(margin):
                    if j<i:polygon=np.empty((0,3));break
                    continue
                polygon=clip(polygon,margin,stats)
                if not len(polygon):break
            if len(polygon):outputs[i]+=area(polygon)/(4*math.pi);stats['area_calls']+=1
    stats['final_gate_cells']=len(regions)
    return {'stratum_masses':mass,'output_masses':outputs.tolist(),'work':dict(stats)}


def main():
    start=time.process_time();deadline=start+110;checks=[];evals=0
    for a,b in ((0.,0.),(1.,2.),(-3.,.5),(10.,-4.),(-1000.,1000.)):
        for lo,hi in ((-math.pi/2,math.pi/2),(-.7,.3),(.1,1.2)):
            def fn(theta):
                c=a*math.cos(theta)+b*math.sin(theta)
                return .5+c/(2*math.sqrt(1+c*c))
            val,err,info=quad(fn,lo,hi,epsabs=1e-12,epsrel=1e-12,full_output=True)
            actual=primitive(hi,(a,b))-primitive(lo,(a,b));evals+=info['neval']
            checks.append(abs(actual-val));assert abs(actual-val)<1e-10
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    state=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    result={'classification':'exact-form three-input forward marginals; numerical angular geometry',
        'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'primitive_checks':len(checks),'primitive_max_error':max(checks),'primitive_scalar_evaluations':evals,
        'rows':[],'random_model_inputs':0}
    for name,model in (('toy',toy),('trained_H2T8_three_steps',trained)):
        clock=time.process_time();forward=push(model,deadline);forwardcpu=time.process_time()-clock
        clock=time.process_time();reference=spherical_reference(model);refcpu=time.process_time()-clock
        outputerror=float(np.max(abs(np.array(forward['output_masses'])-reference['output_masses'])))
        masserror=max(abs(v-reference['stratum_masses'][k]) for k,v in forward['stratum_masses'].items())
        assert outputerror<1e-10 and masserror<1e-10,(name,forward,reference)
        assert abs(sum(forward['stratum_masses'].values())-1)<1e-10
        assert abs(sum(forward['output_masses'])-1)<1e-10
        result['rows'].append({'model':name,'forward':forward,'reference':reference,
            'max_output_error':outputerror,'max_stratum_error':masserror,
            'forward_cpu_seconds':forwardcpu,'reference_cpu_seconds':refcpu})
        (HERE/'third_step_results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'model':name,'output_error':outputerror,'mass_error':masserror,'forward_cpu':forwardcpu,'reference_cpu':refcpu}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-start
    (HERE/'third_step_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
