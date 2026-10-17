"""T77 fixed support-conservation and four-step marginal controls."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from mixed_volume import PHI
from support_volume import SupportOperator

HERE=Path(__file__).resolve().parent

def kernel_control(toy):
    op=SupportOperator(toy,1);op.pending_kind='upper';state=op.empty(np.ones(2))
    state.area[0,0]=1.
    rows=[]
    # Uniform triangle 0<=h0<=h1<=1, density 2. Direct old-coordinate
    # bounds for W=diag(1,2),w=(1,1) provide independent scalar integrals.
    for y in ((.1,.2),(.3,.8),(1.,1.5),(1.5,1.1),(2.,3.),(8.,9.)):
        y=np.array(y);lo=max(-6.,y[0]-1.,y[1]-2.,2*y[0]-y[1]);hi=min(6.,y[0],y[1])
        expected=quad(lambda x:float(PHI(x)),lo,hi,epsabs=1e-13)[0] if hi>lo else 0.
        actual=op.area_density(state,y)
        assert abs(actual-expected)<2e-14,(y,actual,expected)
        rows.append({'y':y.tolist(),'density':actual,'independent_density':expected})
    H,P=op.source_nodes(state)
    assert abs(P.sum()-1)<1e-14 and np.all(P>0) and np.all(H[:,1]>=H[:,0])
    assert abs(np.sum(P*H[:,0])-1/3)<1e-14
    assert abs(np.sum(P*H[:,1])-2/3)<1e-14
    return {'density_checks':rows,'triangle_mass':float(P.sum()),'triangle_nodes':len(P),'work':dict(op.work)}

def main():
    started=time.process_time();deadline=started+110
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    old=json.loads((HERE/'fourth_step_results.json').read_text())
    sourcepaths=[HERE/p for p in ('mixed_volume.py','support_volume.py','check_support_volume.py','fourth_step_results.json')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcepaths}
    result={'classification':'four-input marginal support control, not rank accuracy','sources':sources(),
        'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'independent_support_proof_commit':'ca756aa',
        'random_model_inputs':0,'kernel_control':kernel_control(toy),'rows':[]}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        for n in (16,32):
            baseline=next(r for r in old['rows'] if r['model']==name and r['grid']==n);ref=baseline['reference']
            clock=time.process_time();op=SupportOperator(model,n);state=op.initial();steps=[]
            for t in range(1,5):
                if time.process_time()>deadline:raise TimeoutError('T77 deadline')
                state,output=op.step(state);mass=state.masses()
                assert abs(sum(mass.values())-op.retained**t)<1e-12
                assert all(v>=0 for v in mass.values())
                if name=='toy':assert mass['axis0']==0
                steps.append({'step':t,'support':state.support_kind,'stratum_masses':mass,
                    'mass_defect':sum(mass.values())-op.retained**t})
            expected=['ray','upper','upper','upper'] if name=='toy' else ['ray','upper','full','full']
            assert op.support_history==expected,(name,op.support_history)
            histogram=op.histogram_output(state);mass=state.masses()
            assert abs(histogram.sum()-sum(mass.values()))<1e-12
            row={'model':name,'grid':n,'support_history':op.support_history,'steps':steps,
                'stratum_masses':mass,'integrated_output_masses':output.tolist(),'histogram_output_masses':histogram.tolist(),
                'max_stratum_error':max(abs(v-ref['stratum_masses'][k]) for k,v in mass.items()),
                'max_integrated_output_error':float(np.max(abs(output-ref['output_masses']))),
                'max_histogram_output_error':float(np.max(abs(histogram-ref['output_masses']))),
                'known_input_tail_omission':1-op.retained**4,'mass_error_from_truncated_law':sum(mass.values())-op.retained**4,
                'transport_work':dict(op.work),'transport_cpu_seconds':time.process_time()-clock,
                'old_rectangular_result':baseline}
            result['rows'].append(row)
            print(json.dumps({k:row[k] for k in ('model','grid','max_stratum_error','max_integrated_output_error','transport_cpu_seconds')}),flush=True)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources_after']==result['sources']
    (HERE/'support_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
