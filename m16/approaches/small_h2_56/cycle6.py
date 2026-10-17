import json,time,resource
from pathlib import Path
from math import gamma,pi,factorial
import numpy as np
from geometry import Geometry
from volume import plan
from segment import grouped_mass
from estimator import estimate
from models import load
BASE=Path(__file__).parent;start=time.process_time();out={'checks':[]}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle6.json').write_text(json.dumps(out,indent=2)+'\n')
base,sha=load(5);wi,wh,wo=base
constant=(wi,wh,np.ones_like(wo));row=estimate(constant);row['name']='constant_readout_shortcut';assert row['estimate']==.2;out['checks'].append(row);save()
g=Geometry(20);cells=g.gates(constant);correct=g.correct(constant,cells);rays=g.simplices(correct);orders,bounds=plan(rays,1e-4)
row=grouped_mass(rays,orders,bounds,deadline=start+115);row.update(name='constant_readout_full_geometry',truth=.2,absolute_error=abs(row['estimate']-.2),geometry_counts=g.counts)
assert abs(row['estimate']-.2)<=row['conditional_quadrature_error_bound']+g.counts['omitted_mass_upper_bound']
out['checks'].append(row);save();print(row,flush=True)
for which in ('input','output'):
 model=(wi*1e-16,wh,wo) if which=='input' else (wi,wh,wo*1e-16)
 row=estimate(model);row.update(name=which+'_scale_1e-16',reference=.2792373713559676,absolute_difference=abs(row['estimate']-.2792373713559676))
 assert row['absolute_difference']<row['conditional_quadrature_error_bound']+2.49756e-5
 out['checks'].append(row);save();print(which,row['estimate'],row['absolute_difference'],flush=True)
rays=np.load(BASE/'t6_sigma0.0_cones.npz')['rays'];upper=[];d=6
for begin in range(0,len(rays),65536):
 v=rays[begin:begin+65536];gram=np.swapaxes(v,1,2)@v;a=np.maximum((1-1e-12)/d,gram.min(axis=(1,2))-1e-12)
 upper.extend((gamma(d/2)/(2*pi**(d/2)*factorial(d-1))*np.abs(np.linalg.det(v))*a**(-d/2)).tolist())
upper=np.array(upper);cut=1e-4/len(rays);small=upper<=cut
out['t6_upper_bound_profile']=dict(simplices=len(rays),per_piece_cut=cut,negligible_count=int(small.sum()),omitted_mass_bound=float(upper[small].sum()),retained_count=int((~small).sum()),total_upper_bound=float(upper.sum()))
np.savez_compressed(BASE/'t6_retained_cones.npz',rays=rays[~small],upper=upper[~small]);save();print(out['t6_upper_bound_profile'],flush=True)
