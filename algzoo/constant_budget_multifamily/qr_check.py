"""Smooth synthetic algebra/finite-difference controls before rescue measurements."""
import json
import numpy as np
from qr_rescue import objective,HERE
phi=np.array([[np.sin((i+1)*(j+1)/7)+np.cos((i+3)*(j+2)/11) for j in range(5)] for i in range(31)])
beta=np.arange(15,dtype=float).reshape(5,3)/9-.5
h=np.maximum(np.cos(np.arange(93).reshape(31,3)/13),0)
q,r=np.linalg.qr(phi,mode='reduced');s=q*np.sqrt(31);gamma=r@beta/np.sqrt(31);back=np.linalg.solve(r,gamma*np.sqrt(31));value,g=objective(phi,h,beta);v,gg=objective(s,h,gamma)
err=float(np.max(abs(g-r.T@gg/np.sqrt(31))));coordinate=float(np.max(abs(back-beta)));values=abs(value-v);finite=[];step=1e-6
assert np.min(abs(phi@beta))>1e-4
for i in range(5):
 for j in range(3):
  a=beta.copy();b=beta.copy();a[i,j]+=step;b[i,j]-=step
  finite.append(abs((objective(phi,h,a)[0]-objective(phi,h,b)[0])/(2*step)-g[i,j]))
out=dict(coordinate_error=coordinate,objective_error=values,gradient_transform_error=err,finite_difference_error=max(finite),rank=int(np.linalg.matrix_rank(r)),condition=float(np.linalg.cond(r)))
assert coordinate<1e-12 and values<1e-12 and err<1e-12 and max(finite)<1e-8
(HERE/'qr_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
