"""Event split versus triangulation-only control at fixed model/grid set."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from event_volume import EventInterfaceOperator,EventUniform
from interface_volume import InterfaceOperator

HERE=Path(__file__).resolve().parent

def event_control(model):
    op=EventUniform(model,1,4);op.event_cut=True;op.pending_kind='full';state=op.empty(np.ones(2));state.area[0,0]=1.
    H,P=op.source_nodes(state);negative=H@np.array([-1.,2.])<0
    mass=float(P[negative].sum());moment=P[negative]@H[negative]
    assert abs(mass-.25)<1e-13 and np.max(abs(moment-[1/6,1/24]))<1e-13
    assert abs(P.sum()-1)<1e-13 and np.all(P>0)
    return {'lower_event_mass':mass,'lower_event_moments':moment.tolist(),'work':dict(op.work)}

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
    refsides=json.loads((HERE/'side_mass_results.json').read_text())['rows']
    prior=json.loads((HERE/'interface_results.json').read_text())['rows']
    paths=[HERE/p for p in ('mixed_volume.py','support_volume.py','quadrature_volume.py','affine_volume.py',
        'interface_volume.py','event_volume.py','check_event_volume.py','side_mass_results.json','interface_results.json')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'classification':'event-aligned area quadrature with triangulation-only control','sources':sources(),
        'random_model_inputs':0,'event_control':event_control(toy),'rows':[],'regressions':[]}
    def run(model,n,reconstruction,cut=None):
        clock=time.process_time()
        op=InterfaceOperator(model,n,reconstruction) if cut is None else EventInterfaceOperator(model,n,reconstruction,cut)
        state=op.initial();steps=[]
        for t in range(1,5):
            if time.process_time()>deadline:raise TimeoutError('event quadrature CPU deadline')
            state,output=op.step(state);mass=state.masses()
            assert abs(sum(mass.values())-op.retained**t)<1e-12
            steps.append({'stratum_masses':mass,'output_masses':output.tolist(),'side_masses':[float(s.area.sum()) for s in state.side_states]})
        return {'grid':n,'reconstruction':reconstruction,'event_cut':cut,'steps':steps,'work':op.report(),'cpu_seconds':time.process_time()-clock}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        regression=run(model,16,'affine');old=next(r for r in prior if r['model']==name and r['grid']==16 and r['reconstruction']=='affine')
        difference=max(abs(regression['steps'][t]['stratum_masses'][k]-old['steps'][t]['stratum_masses'][k]) for t in range(4) for k in old['steps'][t]['stratum_masses'])
        assert difference<1e-12
        result['regressions'].append({'model':name,'difference':difference,'run':regression})
        for n in (16,32):
            third=next(r['forward'] for r in refs3 if r['model']==('toy' if name=='toy' else 'trained_H2T8_three_steps'))
            fourth=next(r['reference'] for r in refs4 if r['model']==name and r['grid']==n)
            sides=next(r['reference_side_masses'] for r in refsides if r['model']==name)
            for reconstruction in ('uniform','affine'):
                for cut in (False,True):
                    row=run(model,n,reconstruction,cut);row['model']=name;row['errors']=[]
                    for t,ref in ((3,third),(4,fourth)):
                        error={k:v-ref['stratum_masses'][k] for k,v in row['steps'][t-1]['stratum_masses'].items()}
                        row['errors'].append({'step':t,'signed_stratum_errors':error,'max_stratum_error':max(abs(v) for v in error.values()),
                            'max_output_error':float(np.max(abs(np.array(row['steps'][t-1]['output_masses'])-ref['output_masses'])))})
                    row['signed_step3_side_errors']=(np.array(row['steps'][2]['side_masses'])-sides).tolist()
                    if cut and name=='toy':
                        previous=result['rows'][-1]
                        assert max(abs(row['steps'][t]['stratum_masses'][k]-previous['steps'][t]['stratum_masses'][k]) for t in range(4) for k in row['steps'][t]['stratum_masses'])<1e-12
                    result['rows'].append(row)
                    print(json.dumps({'model':name,'grid':n,'reconstruction':reconstruction,'event_cut':cut,
                        'stage4_error':row['errors'][1]['max_output_error'],'side_errors':row['signed_step3_side_errors'],'cpu':row['cpu_seconds']}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources']==result['sources_after']
    (HERE/'event_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
