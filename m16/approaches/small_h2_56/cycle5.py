import json,time,resource
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from segment import inverse_fifth_segment,grouped_mass
from volume import plan
BASE=Path(__file__).parent;start=time.process_time();out={'primitive':[],'integrals':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle5.json').write_text(json.dumps(out,indent=2)+'\n')
x,w=leggauss(128);x=(x+1)/2;w=w/2
e=np.eye(5)
cases=[(e[0],e[0],'equal',1.),(e[0],e[1],'orthogonal',10/3),(e[0],2*e[0],'collinear',15/64)]
for k in range(12):
 a=np.array([1+((i+1)*(k+2))%7 for i in range(5)],float);a/=a.sum()
 b=np.array([1+((i+3)*(k+1))%11 for i in range(5)],float);b/=b.sum()
 if k%3==0:b=a+(b-a)*1e-12
 cases.append((a,b,'deterministic',None))
for a,b,name,truth in cases:
 p=float(inverse_fifth_segment(a,b));v=(1-x[:,None])*a+x[:,None]*b;reference=float(w@np.sum(v*v,axis=1)**-2.5)
 error=abs(p-reference);assert error<2e-11
 if truth is not None:assert abs(p-truth)<1e-13
 out['primitive'].append(dict(name=name,estimate=p,positive_quadrature=reference,analytic_truth=truth,error=error))
save()
for name in ('t5_cones','t5_sigma0.0025_cones','t5_sigma0.02_cones','t5_partition'):
 rays=np.load(BASE/(name+'.npz'))['rays'];s=time.process_time();orders,bounds=plan(rays,1e-4);planning=time.process_time()-s
 s=time.process_time();row=grouped_mass(rays,orders,bounds,deadline=start+115);row.update(name=name,planning_cpu_seconds=planning,integration_cpu_seconds=time.process_time()-s)
 out['integrals'].append(row);save();print(row,flush=True)
save()
