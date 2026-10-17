"""Frozen same-ray source/positive mass against analytic chamber unions."""
import time
START=time.process_time()
import hashlib,importlib.util,json,math,resource,sys
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/work/admin-t121-rank-mass/m16/approaches/arc_m43_repro')
import estT_snapshot as E
RADIAL=Path('/workspace/work/recover-small_h2_34/m16/approaches/radial_t4_reference/radial.py')
spec=importlib.util.spec_from_file_location('positive_radial',RADIAL);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
class Budget(Exception):pass
def check():
 if time.process_time()-START>55:raise Budget('55 CPU second cap')
class Ops:
 def __init__(self):self.n=0
 def add(self,n):self.n+=n;check()
@lru_cache(None)
def rule(n):
 x,w=np.polynomial.legendre.leggauss(n);x=(x+1)/2;w=w/2;u,v,z=np.meshgrid(x,x,x,indexing='ij');a,b,c=np.meshgrid(w,w,w,indexing='ij');y=np.stack([u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z)],axis=-1).reshape(-1,4);weights=(a*b*c*(1-u)**2*(1-v)).ravel();return y,weights

def positive(rays,n):
 y,w=rule(n);p=y@rays;q=np.sum(p*p,axis=1);return float(abs(np.linalg.det(rays))*np.dot(w,1/(q*q))/(2*math.pi**2))

def cert(rays):
 a=[[F(float(v)) for v in row] for row in rays];g=[[sum(x*y for x,y in zip(u,v)) for v in a] for u in a]
 if min(v for row in g for v in row)>=0:
  det=R.determinant(a);assert det
  return dict(a=str(min(g[i][i] for i in range(4))/4),b=str(max(g[i][i] for i in range(4))),det=str(det),method='exact_acute')
 out=R.bounds(rays);out['method']='exact_KKT';return out

def main():
 geometry=json.loads((HERE/'capture.json').read_text());cache=np.load(HERE/'captured.npz');rays=cache['rays'];ids=cache['parent_ids'];ops=Ops();out=dict(complete=False,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'estT_snapshot.py',HERE/'captured.npz',HERE/'capture.json',RADIAL,HERE/'mass_check.py')},pairs=[],source_named_mass_ops=0,random_network_inputs=0)
 def save():
  out.update(cpu_seconds_including_imports=time.process_time()-START,source_named_mass_ops=ops.n,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  (HERE/'mass_results.json').write_text(json.dumps(out,indent=2)+'\n')
 try:
  for pair in ((0,1),(20,21)):
   parent=geometry['parents'][pair[0]];other=geometry['parents'][pair[1]];assert parent['perm']==other['perm'] and parent['sigma']==other['sigma'];rank=parent['perm'].index(0)+1;pos=F(sum(math.comb(4,k) for k in range(rank,5)),16);truth=(pos if parent['sigma']>0 else 1-pos)/24
   chosen=np.flatnonzero(np.isin(ids,pair));row=dict(parents=list(pair),analytic_mass=str(truth),indices=chosen.tolist(),records=[],rules=[],complete=False);out['pairs'].append(row)
   for i in chosen:
    check();source=E.orthant(rays[i],ops);bound=cert(rays[i]);row['records'].append(dict(index=int(i),source_value=source,radial_certificate=bound))
   row['source_sum']=math.fsum(r['source_value'] for r in row['records']);row['source_error']=row['source_sum']-float(truth);row['negative_source_values']=sum(r['source_value']<0 for r in row['records']);save()
   for n in (20,32,48):
    values=[];bounds=[]
    for entry in row['records']:
     check();i=entry['index'];values.append(positive(rays[i],n));bounds.append(R.probability_bound(entry['radial_certificate'],n)['bound'])
    total=math.fsum(values);error_bound=math.fsum(bounds);row['rules'].append(dict(order=n,sum=total,analytic_error=total-float(truth),conditional_integration_bound=error_bound,values=values,bounds=bounds,nodes=n**3*len(values)));save()
    if error_bound<=1e-9:row['bound_target_met']=True;break
   row.setdefault('bound_target_met',False);row['complete']=True;save()
  out['complete']=True
 except Exception as err:
  out['failure']=repr(err);save()
  if not isinstance(err,Budget):raise
 save();print(json.dumps({'complete':out['complete'],'cpu':out['cpu_seconds_including_imports'],'pairs':[{k:r.get(k) for k in ('parents','analytic_mass','source_sum','source_error','negative_source_values','bound_target_met','complete')}|{'simplices':len(r['records']),'rules':[{k:z[k] for k in ('order','sum','analytic_error','conditional_integration_bound')} for z in r['rules']]} for r in out['pairs']]}))
if __name__=='__main__':main()
