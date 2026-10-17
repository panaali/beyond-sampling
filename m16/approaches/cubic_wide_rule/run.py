"""One fixed cubic wide transform on diagnosed seed0 error geometry."""
import time
START=time.process_time()
import argparse,hashlib,json,math,resource,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'leaf_mass_review';sys.path.insert(0,str(SOURCE));import mass_check as M
E=M.E
class Budget(Exception):pass
class Ops:
 def __init__(self):self.n=0
 def add(self,n):
  self.n+=n
  if time.process_time()-START>55:raise Budget('55CPU cap')
def main(batch):
 assert 0<=batch<8
 ops=Ops();controls=[]
 for name,rays,truth in [('orthant',np.eye(4),1/16),('coordinate-order',np.array([[1.,1.,0,0],[1.,0,0,0],[0,0,1.,0],[0,0,0,1.]]),1/32),('independent-pairs',np.array([[1.,0,0,0],[.5,math.sqrt(3)/2,0,0],[0,0,1.,0],[0,0,.5,math.sqrt(3)/2]]),1/36)]:
  value=E.orthant(rays,ops);controls.append(dict(name=name,rule='original',value=value,truth=truth));assert abs(value-truth)<1e-12
 E._RULES={'wide':E._rule(12,3),'thin':E._rule(48,3)}
 for name,rays,truth in [('orthant',np.eye(4),1/16),('coordinate-order',np.array([[1.,1.,0,0],[1.,0,0,0],[0,0,1.,0],[0,0,0,1.]]),1/32),('independent-pairs',np.array([[1.,0,0,0],[.5,math.sqrt(3)/2,0,0],[0,0,1.,0],[0,0,.5,math.sqrt(3)/2]]),1/36)]:
  value=E.orthant(rays,ops);controls.append(dict(name=name,rule='cubic',value=value,truth=truth));assert abs(value-truth)<1e-12
 controls_ops=ops.n;ops.n=0
 archive=np.load(SOURCE/'error_captured.npz');rays=archive['rays'];parents=archive['parent_ids'];chosen=np.flatnonzero((parents>=12*batch)&(parents<12*(batch+1)));old=np.load(SOURCE/f'error_batch_{batch}.npz');assert np.array_equal(chosen,old['indices']);stem=HERE/f'batch_{batch}'
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'estT_snapshot.py',SOURCE/'error_captured.npz',SOURCE/f'error_batch_{batch}.npz',HERE/'run.py')};rows=[];priorcpu=0.;priorops=0
 if stem.with_suffix('.json').exists():
  previous=json.loads(stem.with_suffix('.json').read_text());assert previous['source_hashes']==hashes;saved=np.load(stem.with_suffix('.npz'));rows=list(zip(saved['indices'].tolist(),saved['values'].tolist(),saved['thin'].tolist(),saved['cutoff'].tolist()));assert [r[0] for r in rows]==chosen[:len(rows)].tolist();priorcpu=previous['cpu'];priorops=previous['mass_named_ops']
  if previous['complete']:print('already complete',batch);return
 out=dict(batch=batch,complete=False,controls=controls,controls_named_ops=controls_ops,source_hashes=hashes,expected=len(chosen))
 def save():
  a=np.asarray(rows,dtype=float).reshape(-1,4);np.savez_compressed(stem.with_suffix('.npz'),indices=a[:,0].astype(int),values=a[:,1],thin=a[:,2].astype(bool),cutoff=a[:,3].astype(bool));out.update(completed=len(rows),sum=math.fsum(r[1] for r in rows),negative_count=sum(r[1]<0 for r in rows),negative_sum=math.fsum(r[1] for r in rows if r[1]<0),thin_count=sum(r[2] for r in rows),determinant_cutoff_count=sum(r[3] for r in rows),cpu=priorcpu+time.process_time()-START,mass_named_ops=priorops+ops.n,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);stem.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
 try:
  for idx in chosen[len(rows):]:
   i=int(idx);v=float(E.orthant(rays[i],ops));assert math.isfinite(v)
   # Diagnostic branch census, separate from estimator arithmetic count; CPU charged.
   B=np.linalg.inv(rays[i].T);S=B@B.T;C=S/np.sqrt(np.outer(np.diag(S),np.diag(S)));thin=bool(max(abs(C[j,k]) for j in range(4) for k in range(j))>.95);cutoff=bool(abs(np.linalg.det(rays[i]))<1e-12)
   rows.append((i,v,thin,cutoff))
   if len(rows)%128==0:save()
  out['complete']=True
 except Exception as err:
  out['failure']=repr(err);save()
  if not isinstance(err,Budget):raise
 save();print(json.dumps({k:v for k,v in out.items() if k not in ('source_hashes','controls')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('batch',type=int);main(p.parse_args().batch)
