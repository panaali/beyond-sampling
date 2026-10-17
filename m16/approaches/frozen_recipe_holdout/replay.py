"""Fixed held-out seed1 captured rays; unchanged original/doubled/positive rules."""
import time
START=time.process_time()
import argparse,hashlib,json,math,resource,sys
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parent;MODE=sys.argv[1];assert MODE in ('exact','recipe108');HERE=TASK/MODE;SOURCE=TASK.parent/'leaf_mass_review';sys.path.insert(0,str(SOURCE));import mass_check as M
E=M.E
class Budget(Exception):pass
class Ops:
 def __init__(self):self.n=0
 def add(self,n):
  self.n+=n
  if time.process_time()-START>55:raise Budget('55CPU cap')
def main(batch):
 assert 0<=batch<8;sourceops=Ops();candidateops=Ops();original=E._RULES;doubled={'wide':E._rule(24,2),'thin':E._rule(96,3)};controls=[]
 for label,rules,o in [('original',original,sourceops),('doubled',doubled,candidateops)]:
  E._RULES=rules
  for name,rays,truth in [('orthant',np.eye(4),1/16),('coordinate-order',np.array([[1.,1.,0,0],[1.,0,0,0],[0,0,1.,0],[0,0,0,1.]]),1/32)]:
   v=float(E.orthant(rays,o));pv=M.positive(rays,32);assert abs(v-truth)<1e-12 and abs(pv-truth)<1e-12;controls.append(dict(rule=label,name=name,value=v,positive=pv,truth=truth))
 controlops=sourceops.n+candidateops.n;sourceops.n=0;candidateops.n=0
 archive=np.load(HERE/'error_captured.npz');rays=archive['rays'];parents=archive['parent_ids'];chosen=np.flatnonzero((parents>=12*batch)&(parents<12*(batch+1)));allowance=1e-9/len(rays);stem=HERE/f'batch_{batch}'
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'estT_snapshot.py',SOURCE/'mass_check.py',HERE/'error_captured.npz',HERE/'error_capture.json',TASK/'replay.py',M.RADIAL)};rows=[];priorcpu=0.;priorold=0;priornew=0
 if stem.with_suffix('.json').exists():
  p=json.loads(stem.with_suffix('.json').read_text());assert p['source_hashes']==hashes;saved=np.load(stem.with_suffix('.npz'));rows=list(zip(*(saved[k].tolist() for k in ('indices','source','doubled','positive','bounds','orders','acute'))));assert [r[0] for r in rows]==chosen[:len(rows)].tolist();priorcpu=p['cpu'];priorold=p['source_mass_named_ops'];priornew=p['doubled_mass_named_ops']
  if p['complete']:print('alreadycomplete',batch);return
 out=dict(mode=MODE,checkpoint_seed=1,batch=batch,complete=False,source_hashes=hashes,controls=controls,control_named_ops=controlops,expected=len(chosen),per_simplex_bound_allowance=allowance)
 def save():
  a=np.asarray(rows,dtype=float).reshape(-1,7);np.savez_compressed(stem.with_suffix('.npz'),indices=a[:,0].astype(int),source=a[:,1],doubled=a[:,2],positive=a[:,3],bounds=a[:,4],orders=a[:,5].astype(int),acute=a[:,6].astype(bool));out.update(completed=len(rows),source_sum=math.fsum(r[1] for r in rows),doubled_sum=math.fsum(r[2] for r in rows),positive_sum=math.fsum(r[3] for r in rows),conditional_bound=math.fsum(r[4] for r in rows),source_negative_count=sum(r[1]<0 for r in rows),doubled_negative_count=sum(r[2]<0 for r in rows),bound_misses=sum(r[4]>allowance for r in rows),source_mass_named_ops=priorold+sourceops.n,doubled_mass_named_ops=priornew+candidateops.n,cpu=priorcpu+time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);stem.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
 try:
  for idx in chosen[len(rows):]:
   i=int(idx);E._RULES=original;old=float(E.orthant(rays[i],sourceops));E._RULES=doubled;candidate=float(E.orthant(rays[i],candidateops));c=M.cert(rays[i])
   for order in (20,32,48):
    bound=M.R.probability_bound(c,order)['bound']
    if bound<=allowance:break
   value=M.positive(rays[i],order);assert all(math.isfinite(v) for v in (old,candidate,value,bound));rows.append((i,old,candidate,value,bound,order,c['method']=='exact_acute'))
   if len(rows)%128==0:save()
  out['complete']=True
 except Exception as err:
  out['failure']=repr(err);save()
  if not isinstance(err,Budget):raise
 finally:E._RULES=original
 save();print(json.dumps({k:v for k,v in out.items() if k not in ('source_hashes','controls')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['exact','recipe108']);p.add_argument('batch',type=int);main(p.parse_args().batch)
