"""Separate quadrature and cross errors on frozen, fully enumerated face grids."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from .algebra import product_weights,direct_integral,weighted_svd,reconstruct,contract
from .check_algebra import author,SOURCE

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons')
WALKER=ROOT/'m16/approaches/sci_tao_v2/core.py'
TRUTH={2:.30773074302315223,4:.47701312599404416}

def main():
    start=time.process_time()
    import torch
    C=author()
    spec=importlib.util.spec_from_file_location('common_corrected_face_oracle',WALKER)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    sources=[SOURCE,WALKER,ROOT/'m16/approaches/sci_tao/core.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out=dict(cases=[],sources=hashes,independent_network_oracle=False,
             numerical_kernels=['weighted TT-SVD','author QR/SVD/pseudoinverse','Gauss-Legendre setup'],
             full_grid_diagnostic_faces=0,full_grid_diagnostic_walk_fpe=0.)
    def save():
        out['cpu_seconds']=time.process_time()-start
        (HERE/'grid_results.json').write_text(json.dumps(out,indent=2)+'\n')
    for H in (2,4):
        checkpoint=ROOT/f'work/checkpoints/H{H}_T4_seed0.pth'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=(state['rnn.weight_ih_l0'].numpy().astype(float).ravel(),
               state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
        _,z23,z12=oracle.tables(4)
        for order in (3,5):
            case=dict(H=H,T=4,order=order,reference_accuracy=TRUTH[H],
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),families=[])
            out['cases'].append(case);save()
            nodes,w=np.polynomial.legendre.leggauss(order);nodes=(nodes+1)/2;w=w/2
            for kind,z in (('23',z23),('12',z12)):
                count=len(oracle.geometry.configs(4,kind));shape=(count,order,order,order)
                indices=np.asarray(list(np.ndindex(shape)));u=np.empty((len(indices),4))
                u[:,0]=(indices[:,0]+.5)/count;u[:,1:]=nodes[indices[:,1:]]
                y,a,b,tau=oracle.face_points(4,kind,u)
                values,costs,cells=oracle.walk_mass(model,y,a,b,tau,kind)
                tensor=values.reshape(shape);cost_tensor=costs.reshape(shape)
                weights=[np.ones(count)/count,w,w,w];exact=direct_integral(tensor,weights)
                family=dict(kind=kind,geometry_factor=z,shape=list(shape),grid_integral=exact,
                    grid_faces=len(values),grid_walk_fpe=float(costs.sum()),grid_cells=int(cells.sum()),fits=[])
                case['families'].append(family)
                out['full_grid_diagnostic_faces']+=len(values)
                out['full_grid_diagnostic_walk_fpe']+=float(costs.sum())
                for ordering,permutation in (('configuration_first',(0,1,2,3)),('configuration_last',(1,2,3,0))):
                    array=tensor.transpose(permutation);weight=[weights[i] for i in permutation]
                    expense=cost_tensor.transpose(permutation);points=np.asarray(list(np.ndindex(array.shape)))
                    for rank in (2,4):
                        if time.process_time()-start>105:raise TimeoutError('T81 CPU guard')
                        queries=set()
                        def evaluate(ps):
                            queries.update(map(tuple,ps))
                            return np.array([array[tuple(p)] for p in ps])
                        fit=C.TensorCross(evaluate,list(array.shape),rank,2)
                        integral=fit.integral(weight);algebra_cost=fit.flops
                        pred=fit.predict(points).reshape(array.shape)
                        direct=direct_integral(pred,weight)
                        assert abs(direct-integral)<1e-10
                        cross_l2=float(np.sqrt(np.sum((pred-array)**2*product_weights(weight))))
                        cores,svd=weighted_svd(array,weight,rank);svd_pred=reconstruct(cores)
                        svd_integral=contract(cores,weight)
                        svd_error=float(np.sqrt(np.sum((svd_pred-array)**2*product_weights(weight))))
                        assert svd_error<=svd['weighted_l2_upper_bound']+1e-12
                        training_cost=float(sum(expense[p] for p in queries))
                        family['fits'].append(dict(ordering=ordering,rank=rank,cross_integral=integral,
                            cross_integral_error=integral-exact,cross_weighted_l2=cross_l2,
                            cross_unique_faces=len(queries),cross_walk_fpe=training_cost,
                            cross_algebra_proxy=algebra_cost,
                            cross_total_fpe_proxy=training_cost+algebra_cost/oracle.fpe_flops(H,4),
                            pivots=fit.conditions,svd_integral=svd_integral,
                            svd_integral_error=svd_integral-exact,svd_weighted_l2=svd_error,
                            svd=svd,svd_uses_full_grid=True))
                        save()
            case['full_grid_accuracy']=1-sum(f['geometry_factor']*f['grid_integral'] for f in case['families'])
            case['full_grid_bias']=case['full_grid_accuracy']-TRUTH[H]
            case['combined_fits']=[]
            for ordering in ('configuration_first','configuration_last'):
                for rank in (2,4):
                    rows=[next(r for r in f['fits'] if r['ordering']==ordering and r['rank']==rank) for f in case['families']]
                    predicted=1-sum(f['geometry_factor']*r['cross_integral'] for f,r in zip(case['families'],rows))
                    svd_predicted=1-sum(f['geometry_factor']*r['svd_integral'] for f,r in zip(case['families'],rows))
                    case['combined_fits'].append(dict(ordering=ordering,rank=rank,accuracy=predicted,
                        bias_to_reference=predicted-TRUTH[H],cross_error_to_full_grid=predicted-case['full_grid_accuracy'],
                        weighted_svd_accuracy=svd_predicted,svd_error_to_full_grid=svd_predicted-case['full_grid_accuracy'],
                        construction_faces=sum(r['cross_unique_faces'] for r in rows),
                        construction_fpe_proxy=sum(r['cross_total_fpe_proxy'] for r in rows)))
            save()
            print(json.dumps({k:case[k] for k in ('H','order','full_grid_accuracy','full_grid_bias','combined_fits')}),flush=True)
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out['complete']=True;save()

if __name__=='__main__':main()
