"""Independent toy densities and full-circle geometric reference."""
from collections import defaultdict
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from two_input import Law

HERE=Path(__file__).resolve().parent


def angular_reference(model):
    wi,wh,wo=model;cuts={0.,math.pi/2,3*math.pi/2,2*math.pi};stats=defaultdict(int)
    def roots(normal):
        if not np.any(normal):return []
        a=math.atan2(-normal[0],normal[1])%(2*math.pi)
        stats['angle_roots']+=2
        return [a,(a+math.pi)%(2*math.pi)]
    for sign in (-1,1):
        a=wh@np.maximum(sign*wi,0)
        for i in (0,1):cuts.update(roots(np.array([sign*a[i],wi[i]])))
    pieces=sorted(cuts);refined=set(cuts)
    for lo,hi in zip(pieces,pieces[1:]):
        theta=(lo+hi)/2;direction=np.array([math.cos(theta),math.sin(theta)])
        sign=1 if direction[0]>0 else -1;a=wh@np.maximum(sign*wi,0)
        Z=np.column_stack((sign*a,wi));gate=Z@direction>0;B=Z*gate[:,None]
        lines=wo@B
        for i in range(len(wo)):
            for j in range(i):
                for root in roots(lines[i]-lines[j]):
                    if lo<root<hi:refined.add(root)
    output=np.zeros(len(wo));mass={k:0. for k in ('origin','axis0','axis1','input_ray','area')}
    pieces=sorted(refined)
    for lo,hi in zip(pieces,pieces[1:]):
        theta=(lo+hi)/2;x=np.array([math.cos(theta),math.sin(theta)])
        h=np.maximum(wi*x[0],0);z=wh@h+wi*x[1];h=np.maximum(z,0)
        stats['validation_direction_network_evaluations']+=1
        sign=1 if x[0]>0 else -1;a=wh@np.maximum(sign*wi,0)
        B=np.column_stack((sign*a,wi))*(z>0)[:,None]
        if not np.any(h):kind='origin'
        elif h[1]==0:kind='axis0'
        elif h[0]==0:kind='axis1'
        elif abs(np.linalg.det(B))>1e-12*np.max(abs(B))**2:kind='area'
        else:kind='input_ray'
        p=(hi-lo)/(2*math.pi);mass[kind]+=p;output[np.argmax(wo@h)]+=p
    return mass,output,dict(stats)


def main():
    start=time.process_time();phi=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    law=Law(toy);expected={'origin':.25+math.atan(.5)/(2*math.pi),'axis0':0.,
        'axis1':(math.atan(2)-math.pi/4)/(2*math.pi),'input_ray':.25,'area':.375}
    assert max(abs(law.masses()[k]-v) for k,v in expected.items())<1e-14
    density_errors=[];integrals=[];calls=0
    for cell in law.cells:
        if cell['rank']==1:
            for z in (.01,.1,.5,1.,2.,4.):
                reference=.5*phi(z) if cell['kind']=='input_ray' else phi(z/math.sqrt(5))/math.sqrt(5)*ndtr(-3*z/math.sqrt(5))
                density_errors.append(abs(law.line_density(cell,z)-reference))
            value,error,info=quad(lambda z:law.line_density(cell,z),0,np.inf,epsabs=1e-12,epsrel=1e-12,full_output=True)
            calls+=info['neval'];integrals.append({'kind':cell['kind'],'value':value,'exact_mass':cell['mass'],'reported_error':error})
            assert abs(value-cell['mass'])<1e-12
        if cell['rank']==2:
            for a,b in ((.1,.2),(.2,1.),(1.,1.5),(2.,3.),(.5,.2)):
                reference=phi(b-a)*phi(2*a-b) if b>a>0 else 0.
                density_errors.append(abs(law.area_density(cell,np.array([a,b]))-reference))
    # Independent integration of the toy area density in x=b-a>0.
    value,error,info=quad(lambda x:phi(x)*ndtr(x),0,np.inf,epsabs=1e-12,full_output=True)
    calls+=info['neval'];assert abs(value-.375)<1e-12
    assert max(density_errors)<1e-14
    rows=[]
    import torch
    checkpoint=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    trained=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    for name,model in (('toy',toy),('trained_H2T8_two_steps',trained),
                       ('duplicate_readout',(trained[0],trained[1],np.ones_like(trained[2])))):
        base=Law(model);baseout=base.output_masses();basemass=base.masses()
        for scale in (1.,1e-16,1e16):
            scaled=(model[0]*scale,model[1],model[2]*scale);current=Law(scaled)
            refmass,refout,refwork=angular_reference(scaled)
            summary=current.summary()
            outputerror=float(np.max(abs(np.array(summary['output_masses'])-refout)))
            masserror=max(abs(summary['stratum_masses'][k]-v) for k,v in refmass.items())
            assert outputerror<1e-12 and masserror<1e-12
            assert max(abs(summary['stratum_masses'][k]-v) for k,v in basemass.items())<1e-12
            assert np.max(abs(np.array(summary['output_masses'])-baseout))<1e-12
            assert abs(sum(summary['stratum_masses'].values())-1)<1e-12
            assert abs(sum(summary['output_masses'])-1)<1e-12
            if name=='duplicate_readout':assert summary['output_masses'][0]>1-1e-12
            rows.append({'name':name,'scale':scale,'representation':summary,'reference_mass':refmass,
                'reference_output':refout.tolist(),'reference_work':refwork,
                'max_stratum_mass_error':masserror,'max_output_error':outputerror})
    result={'classification':'exact sector/stratum formulas with floating angle/density evaluation',
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'toy_masses':expected,
        'toy_density_max_error':max(density_errors),'toy_line_integrals':integrals,
        'adaptive_scalar_evaluations':calls,'toy_density_work':dict(law.work),'rows':rows,
        'random_model_inputs':0,'cpu_seconds_excluding_imports_except_torch':time.process_time()-start}
    (HERE/'two_input_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cpu':result['cpu_seconds_excluding_imports_except_torch'],'toy_density_error':max(density_errors),
        'rows':len(rows),'max_output_error':max(r['max_output_error'] for r in rows)}))

if __name__=='__main__':main()
