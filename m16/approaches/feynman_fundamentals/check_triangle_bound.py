"""Regression for a real T6 cone whose centroid-only Taylor bound failed."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from triangle import _radial_bounds
from triangle_batch import signed_l1_triangles


def main():
    started=time.process_time();here=Path(__file__).resolve().parent
    with np.load(here/'triangle_failure.npz') as data:
        vv=data['endpoints'];outer_order=int(data['order']);local_index=int(data['local_index'])
    actual,stats=signed_l1_triangles(vv,6,absolute_tolerance=1e-11)
    nodes,weights=leggauss(160);nodes=(nodes+1)/2;weights=weights/2
    x,y=np.meshgrid(nodes,nodes,indexing='ij')
    bary=np.array([x,(1-x)*y,(1-x)*(1-y)]).reshape(3,-1)
    ww=(weights[:,None]*weights[None,:]*(1-x)).reshape(-1)
    refs=[];old_bounds=[];new_bounds=[]
    for vertices in vv:
        pts=vertices.T@bary
        refs.append(float(ww@np.sum(pts*pts,axis=0)**-3))
        center=vertices.sum(axis=0);center/=np.linalg.norm(center)
        old_bounds.append(float(np.min(vertices@center))**2)
        new_bounds.append(_radial_bounds(vertices)[0])
    refs=np.asarray(refs)
    result={'fixture':'triangle_failure.npz','original_outer_order':outer_order,
            'failed_local_cone_index':local_index,'global_cone_index':2505+local_index,
            'old_exception_reproduced':'original-triangle quadrature bound unavailable at max_order',
            'triangle_count':len(vv),'signed_l1_residual':float(np.max(np.abs(np.sum(np.abs(vv),axis=2)-1))),
            'minimum_old_q_lower_bound':min(old_bounds),'minimum_new_q_lower_bound':min(new_bounds),
            'requested_absolute_tolerance':1e-11,'max_order':128,
            'max_relative_error_vs_positive160':float(np.max(np.abs(actual/refs-1))),
            'max_absolute_error_vs_positive160':float(np.max(np.abs(actual-refs))),
            'positive_reference_nodes':len(vv)*160**2,'stats':stats,
            'values':actual.tolist(),'reference':refs.tolist(),
            'source_hashes':{name:hashlib.sha256((here/name).read_bytes()).hexdigest()
                             for name in ['triangle.py','triangle_batch.py','triangle_failure.npz']},
            'model_forward_passes':0,'sampled_model_inputs':0,'cpu_seconds':time.process_time()-started}
    (here/'triangle_bound_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('values','reference')},indent=2))


if __name__=='__main__':
    main()
