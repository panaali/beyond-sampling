"""T83 exact synthetic controls and fixed short-horizon marginal comparison."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from affine_volume import AffineOperator

HERE=Path(__file__).resolve().parent

def synthetic(model,kind,gradient):
    op=AffineOperator(model,6);op.pending_kind=kind;state=op.empty(np.array([1.,1.]))
    scalar=.7;gradient=np.array(gradient)
    for i in range(op.n):
        for j in range(op.n):
            volume=op.area_cell_volume(state,i,j)
            if volume>0:
                center=op.centroid(op.area_polygon(state,i,j));state.area[i,j]=volume*(scalar+center@gradient)
    op.prepare(state);max_density_error=0.;max_gradient_error=0.
    for i,j in zip(*np.nonzero(state.area)):
        width=np.array([op.widths[0,i],op.widths[1,j]])
        max_gradient_error=max(max_gradient_error,float(np.max(abs(op.area_gradients[i,j]/width-gradient))))
        p=op.area_polygon(state,i,j)
        predicted=op.area_density_grid[i,j]+((p-op.area_centers[i,j])/width)@op.area_gradients[i,j]
        max_density_error=max(max_density_error,float(np.max(abs(predicted-(scalar+p@gradient)))))
    assert max_density_error<2e-13 and max_gradient_error<2e-13
    H,P=op.source_nodes(state)
    assert abs(P.sum()-state.area.sum())<1e-13 and np.all(P>0)
    # Independent exact first moments: square or upper unit triangle.
    if kind=='full':expected=np.array([scalar/2+gradient[0]/3+gradient[1]/4,scalar/2+gradient[0]/4+gradient[1]/3])
    else:expected=np.array([scalar/6+gradient[0]/12+gradient[1]/8,scalar/3+gradient[0]/8+gradient[1]/4])
    moment_error=float(np.max(abs(P@H-expected)));assert moment_error<2e-13
    # Constant density must reconstruct with identically zero gradients.
    for i,j in zip(*np.nonzero(state.area)):state.area[i,j]=op.area_cell_volume(state,i,j)
    op.prepare(state);assert np.max(abs(op.area_gradients))<2e-13
    return {'support':kind,'gradient':gradient.tolist(),'max_density_error':max_density_error,'max_gradient_error':max_gradient_error,
        'moment_error':moment_error,'work':dict(op.work)}

def axis_control(model):
    op=AffineOperator(model,6);state=op.empty(np.ones(2));state.support_kind='full'
    intercept=np.array([.7,.8]);slope=np.array([.3,-.2])
    for axis in (0,1):
        edge=state.edges[axis];mid=(edge[:-1]+edge[1:])/2
        state.axes[axis]=np.diff(edge)*(intercept[axis]+slope[axis]*mid)
    H,P=op.source_nodes(state)
    error=float(np.max(abs(P@H-(intercept/2+slope/3))))
    assert error<2e-13
    assert abs(P.sum()-sum(intercept+slope/2))<2e-13
    return {'axis_first_moment_error':error,'work':dict(op.work)}

def main():
    started=time.process_time();deadline=started+110
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth')
    data=torch.load(path,map_location='cpu',weights_only=True)
    trained=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    trained=(trained[0].reshape(-1),trained[1],trained[2])
    toy=(np.ones(2),np.diag([1.,2.]),np.array([[0.,0.],[2.,-1.],[-1.,2.],[.5,.5]]))
    ref3=json.loads((HERE/'third_step_results.json').read_text())['rows']
    old=json.loads((HERE/'quadrature_results.json').read_text())['rows']
    ref4=json.loads((HERE/'fourth_step_results.json').read_text())['rows']
    paths=[HERE/s for s in ('mixed_volume.py','support_volume.py','quadrature_volume.py','affine_volume.py',
        'check_affine_volume.py','third_step_results.json','quadrature_results.json','fourth_step_results.json')]
    sources=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result={'classification':'mass-only positive local affine reconstruction, marginal controls only',
        'sources':sources(),'random_model_inputs':0,'synthetic_controls':[synthetic(toy,k,g) for k in ('full','upper') for g in ((.2,.3),(.2,-.3))],
        'axis_control':axis_control(toy),
        'rows':[],'coordinate_controls':[]}
    def run(model,n):
        clock=time.process_time();op=AffineOperator(model,n);state=op.initial();steps=[]
        for t in range(1,5):
            if time.process_time()>deadline:raise TimeoutError('T83 CPU deadline')
            state,output=op.step(state);mass=state.masses()
            assert abs(sum(mass.values())-op.retained**t)<1e-12
            steps.append({'stratum_masses':mass,'output_masses':output.tolist()})
        histogram=op.histogram_output(state)
        return {'grid':n,'steps':steps,'histogram_output_masses':histogram.tolist(),
            'work':dict(op.work),'cpu_seconds':time.process_time()-clock}
    for name,model in (('toy',toy),('trained_H2T8',trained)):
        for n in (16,32):
            row=run(model,n);row['model']=name
            third=next(r['forward'] for r in ref3 if r['model']==('toy' if name=='toy' else 'trained_H2T8_three_steps'))
            fourth=next(r['reference'] for r in ref4 if r['model']==name and r['grid']==n)
            row['errors']=[]
            for t,ref in ((3,third),(4,fourth)):
                masses=row['steps'][t-1]['stratum_masses'];output=np.array(row['steps'][t-1]['output_masses'])
                errors={k:v-ref['stratum_masses'][k] for k,v in masses.items()}
                row['errors'].append({'step':t,'signed_stratum_errors':errors,'max_stratum_error':max(abs(v) for v in errors.values()),
                    'max_output_error':float(np.max(abs(output-ref['output_masses'])))})
            row['old_uniform_order4']=next(r for r in old if r['model']==name and r['grid']==n and r['source_order']==4)
            row['uniform_histogram_readout_error']=float(np.max(abs(np.array(row['histogram_output_masses'])-fourth['output_masses'])))
            result['rows'].append(row)
            print(json.dumps({'model':name,'grid':n,'errors':row['errors'],'cpu':row['cpu_seconds']}),flush=True)
        D=np.diag([2.,.5]);inverse=np.linalg.inv(D)
        changed=run((D@model[0],D@model[1]@inverse,model[2]@inverse),16)
        base=next(r for r in result['rows'] if r['model']==name and r['grid']==16)
        discrepancy=max(abs(changed['steps'][t]['stratum_masses'][k]-base['steps'][t]['stratum_masses'][k])
            for t in range(4) for k in base['steps'][t]['stratum_masses'])
        discrepancy=max(discrepancy,max(float(np.max(abs(np.array(changed['steps'][t]['output_masses'])-base['steps'][t]['output_masses']))) for t in range(4)))
        assert discrepancy<1e-12
        changed.update(model=name,max_coordinate_discrepancy=discrepancy);result['coordinate_controls'].append(changed)
    result['cpu_seconds_including_torch_import']=time.process_time()-started
    result['sources_after']=sources();assert result['sources_after']==result['sources']
    (HERE/'affine_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_including_torch_import']}))

if __name__=='__main__':main()
