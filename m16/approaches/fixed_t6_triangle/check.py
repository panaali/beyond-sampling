"""Frozen same-endpoint controls; no stochastic network inputs."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/clean-neurons');sys.path.insert(0,str(ROOT))
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit
from m16.approaches.small_ladder_review.portable_cost.t6.triangle import Engine
from m16.approaches.small_ladder_review.portable_cost.t6.jacobi import outer_rule
from primitive import prepare,evaluate
HERE=Path(__file__).resolve().parent

def main():
 start=time.process_time(); rows=[]; out=dict(rows=rows,model_forwards=0,sampled_model_inputs=0)
 def save():
  out['cpu_seconds']=time.process_time()-start
  (HERE/'control_results.json').write_text(json.dumps(out,indent=2)+'\n')
 def compare(label,base,prefix,rest,reference=False):
  a=Audit();old=Engine(a); b=Audit();new=Engine(b)
  vertices=prefix[:,None,:]+rest[:,None,None]*base[None,:,:]
  t=time.process_time();v0,s0=old.batch(vertices);t0=time.process_time()-t
  t=time.process_time();frame=prepare(base);v1,s1=evaluate(prefix,rest,frame,new.positive);t1=time.process_time()-t
  rel=float(np.max(np.abs(v1/v0-1)));absolute=float(np.max(np.abs(v1-v0)))
  row=dict(label=label,n=len(rest),relative_difference=rel,absolute_difference=absolute,old_cpu=t0,new_cpu=t1,old_stats=s0,new_stats=s1,setup_frames=1,
           fallback_count_same=s0['positive_quadrature_fallbacks']==s1['positive_quadrature_fallbacks'])
  if reference:
   refs=np.array([new.positive(v,6,1e-11,128)[0] for v in vertices])
   row.update(old_positive_reference_relative=float(np.max(np.abs(v0/refs-1))),new_positive_reference_relative=float(np.max(np.abs(v1/refs-1))))
  rows.append(row);save()
  if rel>1e-9 or not row['fallback_count_same']:out['diagnosis_required']=True
  return v0,v1
 try:
  eye=np.eye(3,6)
  for width in (1.,.1,.01,.001,.0001,.00001,.0000001,0.):
   base=np.ones((3,6))+width*eye;base/=base.sum(axis=1)[:,None]
   compare(['symmetric',width],base,np.zeros((1,6)),np.ones(1),True)
  anchor=.2*np.ones(6)/6+.8*np.eye(6)[0]
  for width in (.1,.01,.001,.0001,.00001,.0000001,0.):
   base=np.stack([anchor,anchor+width*(eye[1]-eye[0]),anchor+width*(eye[2]-eye[0])])
   compare(['asymmetric',width],base,np.zeros((1,6)),np.ones(1),True)
  compare('zero_remainder',eye,anchor[None,:],np.zeros(1),True)
  # Same exact zero-coalescence endpoints with a nonsymmetric fixed frame.
  fixture=HERE.parent/'directed_chebyshev/t6_fixture.json';rays=np.array(json.loads(fixture.read_text())['rays'])
  out['fixture_sha256']=hashlib.sha256(fixture.read_bytes()).hexdigest()
  for n in (4,8):
   y,rest,w=outer_rule(n,Audit())
   for index,ray in enumerate(rays):compare(['stored',index,n],ray[:,3:].T,y@ray[:,:3].T,rest)
  # Fixed ordinary cone and 4096 nodes, three full cold repetitions, paired order.
  ray=np.eye(6);y,rest,w=outer_rule(16,Audit());prefix=y@ray[:,:3].T
  for i in range(3):compare(['ordinary4096',i],ray[:,3:].T,prefix,rest)
  out['max_relative_difference']=max(r['relative_difference'] for r in rows)
  out['changed_fallback_rows']=[r['label'] for r in rows if not r['fallback_count_same']]
  out['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'primitive.py',ROOT/'m16/approaches/feynman_fundamentals/triangle_batch.py',ROOT/'m16/approaches/small_ladder_review/portable_cost/t6/triangle.py']}
 except Exception as exc:
  out['failure']=dict(type=type(exc).__name__,message=str(exc));save();raise
 save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','source_hashes')},indent=2))
if __name__=='__main__':main()
