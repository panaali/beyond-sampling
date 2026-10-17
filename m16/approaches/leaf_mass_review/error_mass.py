"""Fixed twelve-parent checkpoints for captured classification-error mass."""
import time
START=time.process_time()
import argparse,hashlib,json,math,resource
from pathlib import Path
import numpy as np
import mass_check as M
HERE=Path(__file__).resolve().parent
class Budget(Exception):pass
class Ops:
 def __init__(self):self.n=0
 def add(self,n):
  self.n+=n
  if time.process_time()-START>55:raise Budget('55 CPU cap; remaining values unresolved')
def main(batch):
 assert 0<=batch<8
 archive=np.load(HERE/'error_captured.npz');rays=archive['rays'];parents=archive['parent_ids'];chosen=np.flatnonzero((parents>=12*batch)&(parents<12*(batch+1)));allowance=1e-9/len(rays);stem=HERE/f'error_batch_{batch}';hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'estT_snapshot.py',HERE/'error_captured.npz',HERE/'error_capture.json',HERE/'mass_check.py',HERE/'error_mass.py',M.RADIAL)}
 rows=[];prior_cpu=0.;prior_ops=0;prior_reused=0
 if stem.with_suffix('.json').exists():
  previous=json.loads(stem.with_suffix('.json').read_text());assert previous['source_hashes']==hashes
  saved=np.load(stem.with_suffix('.npz'));rows=list(zip(saved['indices'].tolist(),saved['source'].tolist(),saved['positive'].tolist(),saved['bounds'].tolist(),saved['orders'].tolist(),saved['acute'].tolist(),saved['reused'].tolist()));prior_cpu=previous['cumulative_cpu'];prior_ops=previous['cumulative_source_named_ops'];prior_reused=previous['reused_selected_controls']
  assert [r[0] for r in rows]==chosen[:len(rows)].tolist()
  if previous['complete']:print('already complete',batch);return
 ops=Ops();out=dict(kind='error_mass',batch=batch,parent_start=12*batch,parent_stop=12*(batch+1),expected_simplices=len(chosen),per_simplex_bound_allowance=allowance,source_hashes=hashes,complete=False)
 def save():
  array=np.asarray(rows,dtype=float).reshape(-1,7);np.savez_compressed(stem.with_suffix('.npz'),indices=array[:,0].astype(int),source=array[:,1],positive=array[:,2],bounds=array[:,3],orders=array[:,4].astype(int),acute=array[:,5].astype(bool),reused=array[:,6].astype(bool))
  out.update(completed_simplices=len(rows),source_sum=math.fsum(r[1] for r in rows),positive_sum=math.fsum(r[2] for r in rows),conditional_integration_bound=math.fsum(r[3] for r in rows),negative_source_count=sum(r[1]<0 for r in rows),negative_source_sum=math.fsum(r[1] for r in rows if r[1]<0),bound_target_misses=sum(r[3]>allowance for r in rows),reused_selected_controls=sum(r[6] for r in rows),current_cpu=time.process_time()-START,cumulative_cpu=prior_cpu+time.process_time()-START,cumulative_source_named_ops=prior_ops+ops.n,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  stem.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
 try:
  for idx in chosen[len(rows):]:
   ops.add(0);i=int(idx);reuse=False
   source=float(M.E.orthant(rays[i],ops));c=M.cert(rays[i])
   for order in (20,32,48):
    bound=M.R.probability_bound(c,order)['bound']
    if bound<=allowance:break
   value=M.positive(rays[i],order)
   rows.append((i,float(source),float(value),float(bound),int(order),c['method']=='exact_acute',reuse))
   if len(rows)%128==0:save()
  out['complete']=True
 except Exception as err:
  out['failure']=repr(err);save()
  if not isinstance(err,Budget):raise
 save();print(json.dumps({k:v for k,v in out.items() if k!='source_hashes'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('batch',type=int);main(p.parse_args().batch)
