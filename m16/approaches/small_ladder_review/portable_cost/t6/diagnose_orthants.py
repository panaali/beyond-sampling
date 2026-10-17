"""Measure strict orthant discrepancy; no projection is used for integration."""
import json,math,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;start=time.process_time()
rays=np.load(HERE/'independent_correct_rays.npz')['rays']
dominant=np.take_along_axis(rays,np.argmax(abs(rays),axis=2)[:,:,None],axis=2)[:,:,0]
signs=np.where(dominant<0,-1.,1.)
signed=rays*signs[:,:,None];mixed=np.any(signed<0,axis=(1,2));wrong=np.maximum(-signed,0)
projected=signs[:,:,None]*np.maximum(signed,0);norms=np.sum(abs(projected),axis=1)
assert np.all(norms>0)
projected/=norms[:,None,:]
column_errors=np.sqrt(np.sum((rays-projected)**2,axis=1));eps=column_errors.max(axis=1)
radius=1/math.sqrt(6)-eps;assert radius.min()>0
# Use det(projected)<=1 to avoid a determinant kernel in this diagnosis.
bound=(column_errors.sum(axis=1)/radius**6+6*eps/radius**7)/(120*math.pi**3)
out=dict(mixed_orthant_cones=int(mixed.sum()),total_cones=len(rays),largest_wrong_signed_coordinate=float(wrong.max()),maximum_column_l2_shift=float(eps.max()),sum_column_l2_shifts=float(column_errors.sum()),conservative_mass_shift_bound_sum=float(bound.sum()),maximum_per_cone_bound=float(bound.max()),projected_l1_error=float(np.max(abs(np.sum(abs(projected),axis=1)-1))),projected_mixed_orthant_cones=int(np.sum(np.any((projected.min(axis=2)<0)&(projected.max(axis=2)>0),axis=1))),mass_evaluations=0,projection_used_by_estimator=False,cpu_seconds=time.process_time()-start,qualification='Analytic projection perturbation bound in real arithmetic; computed magnitudes and original floating topology are not interval-certified.')
(HERE/'orthant_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
