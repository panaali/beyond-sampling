import json,time,resource
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from volume import bernstein_bound,supporting_lower
from triangle_outer import grouped_mass,rule,endpoints,signed_l1_triangles
from segment6 import grouped_mass as segments
BASE=Path(__file__).parent;start=time.process_time();out={'cases':[],'inner_checks':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle11.json').write_text(json.dumps(out,indent=2)+'\n')
for name in ('identity','t6_partition','t6_retained_cones'):
 if name=='identity':rays=np.array([np.eye(6)]);idx=np.array([0])
 else:
  allrays=np.load(BASE/(name+'.npz'))['rays'];rank=np.argsort(np.abs(np.linalg.det(allrays)));idx=np.unique(np.r_[np.linspace(0,len(allrays)-1,24,dtype=int),rank[np.linspace(0,len(rank)-1,8,dtype=int)]]);rays=allrays[idx]
 if name!='identity':
  yy,rr,ww=rule(6,8);x,w=leggauss(48);x=(x+1)/2;w=w/2;u,v=np.meshgrid(x,x,indexing='ij');bary=np.stack([u.ravel(),((1-u)*v).ravel(),((1-u)*(1-v)).ravel()],axis=1);weight=(w[:,None]*w[None,:]*(1-u)).ravel()
  for k in np.linspace(0,len(rays)-1,4,dtype=int):
   tri=endpoints(rays[k],yy,rr)[len(yy)//2];value,stats=signed_l1_triangles(tri)
   points=bary@tri;ref=float(weight@np.sum(points*points,axis=1)**-3)
   out['inner_checks'].append(dict(name=name,cone_index=int(idx[k]),estimate=float(value),direct_positive_reference=ref,relative_difference=float(abs(value-ref)/ref),fallbacks=stats['positive_quadrature_fallbacks']));save()
 for n in (4,8):
  bounds=np.array([bernstein_bound(v,n,degree=n-1,radial_lower=supporting_lower(v)) for v in rays]);orders=np.full(len(rays),n)
  row=dict(name=name,order=n,indices=idx.tolist(),triangle=grouped_mass(rays,orders,bounds,deadline=start+115),segment=segments(rays,orders,bounds,deadline=start+115));out['cases'].append(row);save();print(name,n,row['triangle'],flush=True)
save()
