"""Frozen T77 source-order and coordinate diagnostics; no input sampling."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from quadrature_volume import QuadratureOperator

HERE=Path(__file__).resolve().parent

def main():
    started=time.process_time();deadline=started+20
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    old=json.loads((HERE/'support_results.json').read_text())
    refs2=json.loads((HERE/'two_input_results.json').read_text())['rows']
    refs3=json.loads((HERE/'third_step_results.json').read_text())['rows']
    paths=[HERE/s for s in ('mixed_volume.py','support_volume.py','quadrature_volume.py','check_quadrature_volume.py',
        'support_results.json','two_input_results.json','third_step_results.json')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'sources':sources(),'classification':'source quadrature versus uniform projection diagnostic',
        'random_model_inputs':0,'rows':[],'coordinate_controls':[]}
    def run(model,n,order):
        clock=time.process_time();op=QuadratureOperator(model,n,order);state=op.initial();steps=[]
        for t in range(1,5):
            if time.process_time()>deadline:raise TimeoutError('T77 short control CPU deadline')
            state,output=op.step(state);steps.append(state.masses())
            assert abs(sum(steps[-1].values())-op.retained**t)<1e-12
        histogram=op.histogram_output(state)
        return {'model':None,'grid':n,'source_order':order,'steps':steps,'output_masses':output.tolist(),
            'histogram_output_masses':histogram.tolist(),'support_history':op.support_history,
            'work':dict(op.work),'cpu_seconds':time.process_time()-clock}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        for n in (16,32):
            baseline=next(r for r in old['rows'] if r['model']==name and r['grid']==n)
            ref4=baseline['old_rectangular_result']['reference']
            shortname='toy' if name=='toy' else 'trained_H2T8_two_steps'
            ref2=next(r['representation'] for r in refs2 if r['name']==shortname and r['scale']==1.)
            ref3=next(r['forward'] for r in refs3 if r['model']==('toy' if name=='toy' else 'trained_H2T8_three_steps'))
            for order in (2,4):
                row=run(model,n,order);row['model']=name
                if order==2:
                    discrepancy=max(abs(row['steps'][-1][k]-v) for k,v in baseline['stratum_masses'].items())
                    discrepancy=max(discrepancy,float(np.max(abs(np.array(row['output_masses'])-baseline['integrated_output_masses']))))
                    row['old_order2_implementation_discrepancy']=discrepancy;assert discrepancy<1e-12
                row['stage_stratum_errors']=[max(abs(row['steps'][t-1][k]-v) for k,v in ref['stratum_masses'].items())
                    for t,ref in ((2,ref2),(3,ref3),(4,ref4))]
                row['signed_stage_stratum_errors']=[{k:row['steps'][t-1][k]-v for k,v in ref['stratum_masses'].items()}
                    for t,ref in ((2,ref2),(3,ref3),(4,ref4))]
                row['endpoint_error']=float(np.max(abs(np.array(row['output_masses'])-ref4['output_masses'])))
                row['histogram_error']=float(np.max(abs(np.array(row['histogram_output_masses'])-ref4['output_masses'])))
                result['rows'].append(row)
                print(json.dumps({k:row[k] for k in ('model','grid','source_order','stage_stratum_errors','endpoint_error','cpu_seconds')}),flush=True)
        for change,D in (('swap',np.array([[0.,1.],[1.,0.]])),('diagonal',np.diag([2.,.5]))):
            inv=np.linalg.inv(D);modified=(D@model[0],D@model[1]@inv,model[2]@inv)
            for order in (2,4):
                row=run(modified,16,order);row['model']=name;row['change']=change
                base=next(r for r in result['rows'] if r['model']==name and r['grid']==16 and r['source_order']==order)
                axis_map={'axis0':'axis1','axis1':'axis0'} if change=='swap' else {}
                row['max_stratum_coordinate_discrepancy']=max(abs(row['steps'][t][k]-base['steps'][t][axis_map.get(k,k)])
                    for t in range(4) for k in row['steps'][t])
                row['max_output_coordinate_discrepancy']=float(np.max(abs(np.array(row['output_masses'])-base['output_masses'])))
                if change=='diagonal':assert max(row['max_stratum_coordinate_discrepancy'],row['max_output_coordinate_discrepancy'])<1e-12
                expected=[{'upper':'lower','lower':'upper'}.get(s,s) for s in base['support_history']] if change=='swap' else base['support_history']
                assert row['support_history']==expected
                result['coordinate_controls'].append(row)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources']==result['sources_after']
    (HERE/'quadrature_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
