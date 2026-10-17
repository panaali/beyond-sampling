"""T91 fixed controls and comparison; no source histories or input sampling."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from interface_volume import InterfaceOperator

HERE=Path(__file__).resolve().parent

def synthetic(model,reconstruction):
    op=InterfaceOperator(model,8,reconstruction);op.pending_kind='full';state=op.empty(np.ones(2))
    for side,rho in zip(state.side_states,(2.,1.)):side.area[:]=rho*side.cell_volumes
    state.area=state.side_states[0].area+state.side_states[1].area
    H,P=op.source_nodes(state)
    assert np.all(P>0) and abs(P.sum()-1.5)<1e-12
    error=float(np.max(abs(P@H-[2/3,5/6])));assert error<1e-12
    return {'reconstruction':reconstruction,'moment_error':error,'work':op.report()}

def thin_control(reconstruction):
    model=(np.array([1.,.999]),np.diag([1.,2.]),np.array([[1.,0.],[0.,1.]]))
    op=InterfaceOperator(model,8,reconstruction);op.pending_kind='full';state=op.empty(np.ones(2))
    for side in state.side_states:side.area[:]=side.cell_volumes
    # Remove one cell entirely on each side; reconstruction must not refill it.
    removed=[]
    for side in state.side_states:
        occupied=list(zip(*np.nonzero(side.area)));index=occupied[len(occupied)//2]
        side.area[index]=0.;removed.append(index)
    state.area=state.side_states[0].area+state.side_states[1].area
    H,P=op.source_nodes(state)
    assert np.all(np.isfinite(H)) and np.all(np.isfinite(P)) and np.all(P>0)
    assert abs(P.sum()-state.area.sum())<1e-12
    for side,index in zip(state.side_states,removed):assert side.area[index]==0
    positive=np.concatenate([s.cell_volumes[s.cell_volumes>0] for s in state.side_states])
    return {'reconstruction':reconstruction,'minimum_positive_polygon_area':float(positive.min()),
        'mass':float(P.sum()),'work':op.report()}

def main():
    started=time.process_time();deadline=started+110
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    ref3=json.loads((HERE/'third_step_results.json').read_text())['rows']
    ref4=json.loads((HERE/'fourth_step_results.json').read_text())['rows']
    uniform=json.loads((HERE/'quadrature_results.json').read_text())['rows']
    affine=json.loads((HERE/'affine_results.json').read_text())['rows']
    paths=[HERE/s for s in ('mixed_volume.py','support_volume.py','quadrature_volume.py','affine_volume.py',
        'interface_volume.py','check_interface_volume.py','third_step_results.json','fourth_step_results.json',
        'quadrature_results.json','affine_results.json')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'classification':'two-sided area interface, marginal transport only','sources':sources(),'random_model_inputs':0,
        'synthetic_controls':[synthetic(toy,r) for r in ('uniform','affine')],
        'thin_zero_cell_controls':[thin_control(r) for r in ('uniform','affine')],'rows':[]}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        for n in (16,32):
            third=next(r['forward'] for r in ref3 if r['model']==('toy' if name=='toy' else 'trained_H2T8_three_steps'))
            fourth=next(r['reference'] for r in ref4 if r['model']==name and r['grid']==n)
            for reconstruction in ('uniform','affine'):
                clock=time.process_time();op=InterfaceOperator(model,n,reconstruction);state=op.initial();steps=[]
                for t in range(1,5):
                    if time.process_time()>deadline:raise TimeoutError('T91 CPU deadline')
                    state,output=op.step(state);mass=state.masses()
                    assert abs(sum(mass.values())-op.retained**t)<1e-12
                    sides=[float(s.area.sum()) for s in state.side_states]
                    assert abs(sum(sides)-mass['area'])<1e-12
                    if name=='toy':assert sides[1]==0 and mass['axis0']==0
                    steps.append({'stratum_masses':mass,'output_masses':output.tolist(),'side_masses':sides})
                old=next(r for r in (uniform if reconstruction=='uniform' else affine) if r['model']==name and r['grid']==n and r.get('source_order',4)==4)
                oldsteps=[{'stratum_masses':s} for s in old['steps']] if reconstruction=='uniform' else old['steps']
                invariance=max(abs(steps[t]['stratum_masses'][k]-oldsteps[t]['stratum_masses'][k])
                    for t in range(4 if name=='toy' else 3) for k in steps[t]['stratum_masses'])
                assert invariance<1e-12,(name,n,reconstruction,invariance)
                errors=[]
                for t,ref in ((3,third),(4,fourth)):
                    difference={k:v-ref['stratum_masses'][k] for k,v in steps[t-1]['stratum_masses'].items()}
                    errors.append({'step':t,'signed_stratum_errors':difference,'max_stratum_error':max(abs(v) for v in difference.values()),
                        'max_output_error':float(np.max(abs(np.array(steps[t-1]['output_masses'])-ref['output_masses'])))})
                row={'model':name,'grid':n,'reconstruction':reconstruction,'steps':steps,'errors':errors,
                    'required_unsplit_invariance_error':invariance,'work':op.report(),'cpu_seconds':time.process_time()-clock}
                result['rows'].append(row)
                print(json.dumps({'model':name,'grid':n,'reconstruction':reconstruction,'errors':errors,'cpu':row['cpu_seconds']}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources_after']==result['sources']
    (HERE/'interface_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
