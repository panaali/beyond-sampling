"""Deterministic geometry checks and a bounded batch benchmark."""
import json
import time
from pathlib import Path
import numpy as np
from triangle import triangle_inverse_norm
from triangle_batch import signed_l1_triangles


def main():
    started=time.process_time();rows=[]
    for d in (4,5,6):
        base=np.eye(3,d)
        cases=[]
        labels=[]
        for width in (1.,.1,.01,.001,.0001,.00001,.0000001,0.):
            v=np.ones((3,d))+width*base
            cases.append(v/np.sum(v,axis=1)[:,None])
            labels.append(['symmetric',width])
        for width in (.1,.01,.001,.0001,.00001,.0000001,0.):
            anchor=.2*np.ones(d)/d+.8*np.eye(d)[0]
            cases.append(np.stack([anchor,anchor+width*(base[1]-base[0]),
                                   anchor+width*(base[2]-base[0])]))
            labels.append(['asymmetric',width])
        cases=np.array(cases)
        actual,stats=signed_l1_triangles(cases,d)
        refs=[];ref_stats=[]
        for v in cases:
            value,meta=triangle_inverse_norm(v,d,known_plane_normal=np.ones(d),
                                            min_singular_ratio=2)
            refs.append(value);ref_stats.append(meta)
        permuted,perm_stats=signed_l1_triangles(cases[:,[2,0,1]],d)
        rows.append({'dimension':d,'case_labels':labels,'values':actual.tolist(),'positive_reference':refs,
                     'relative_errors':np.abs(actual/refs-1).tolist(),
                     'max_permutation_difference':float(np.max(np.abs(permuted/actual-1))),
                     'stats':stats,'permutation_stats':perm_stats,
                     'reference_fallback_nodes':sum(s['quadrature_nodes'] for s in ref_stats)})
    # Deterministic ordinary geometry; no random network inputs or RNG.
    n=4096;d=6
    grid=np.arange(n,dtype=float)
    weights=np.stack([.05+.1*(grid%17)/17,.1+.15*(grid%31)/31,.05+.2*(grid%13)/13],axis=1)
    batch=np.eye(3,d)[None,:,:]+weights[:,:,None]
    batch/=batch.sum(axis=2)[:,:,None]
    t=time.process_time(); values,stats=signed_l1_triangles(batch,d);batch_cpu=time.process_time()-t
    t=time.process_time();refs=[triangle_inverse_norm(v,d,known_plane_normal=np.ones(d))[0] for v in batch[:32]]
    scalar_cpu=time.process_time()-t
    result={'width_sweeps':rows,'benchmark_triangles':n,'benchmark_cpu_seconds':batch_cpu,
            'benchmark_stats':stats,'scalar_reference_count':32,'scalar_reference_cpu_seconds':scalar_cpu,
            'benchmark_reference_relative_error':float(np.max(np.abs(values[:32]/refs-1))),
            'model_forward_passes':0,'sampled_model_inputs':0,
            'cpu_seconds':time.process_time()-started}
    Path(__file__).with_name('triangle_batch_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='width_sweeps'},indent=2))
    print('width_sweep_max_relative_errors', [max(x['relative_errors']) for x in rows])


if __name__=='__main__':
    main()
