"""Independent product-Gaussian falsifier for semi-infinite quadrature."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
start=time.process_time()
path=Path('/workspace/work/recover-small_h2_910/m16/approaches/endpoint_cumulant_review/full_covariance.py')
spec=importlib.util.spec_from_file_location('tail_review_endpoint',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rows=[]
for a in (0.,8.,40.,100.,1000.):
    stats=dict(quad_calls=0,quad_evaluations=0,quad_error_estimate_sum=0.,singular_limits=0,
               outside_unit_corrections=[],singular_orthant_error_bounds=[])
    row=dict(mean=[a,0.],expected=float(ndtr(a)*.5))
    try:
        value=module.orthant([a,0.],np.eye(2),[False,False],1e-12,stats)
        row.update(value=value,error=value-row['expected'],stats=stats)
    except Exception as e:row['failure']=repr(e)
    rows.append(row)
out=dict(rows=rows,cpu_seconds=time.process_time()-start,
         scope='Independent analytic rho0 product reference, no trained-model claim.')
Path(__file__).with_name('gaussian_tail_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
