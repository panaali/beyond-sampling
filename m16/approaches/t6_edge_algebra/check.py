"""Frozen raw/normalized alternatives, identical endpoints and no model calls."""
import hashlib,importlib.util,itertools,json,sys,time
from pathlib import Path
import numpy as np
from primitive import prepare,evaluate
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons');sys.path.insert(0,str(ROOT))
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit
from m16.approaches.small_ladder_review.portable_cost.t6.triangle import Engine
from m16.approaches.small_ladder_review.portable_cost.t6.jacobi import outer_rule
old_path=HERE.parent/'fixed_t6_triangle/primitive.py'
spec=importlib.util.spec_from_file_location('old_frame',old_path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def main():
 start=time.process_time();out=dict(rows=[],source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path,HERE/'primitive.py')},model_forward_passes=0)
 def save():
  out['cpu_seconds']=time.process_time()-start;(HERE/'control_results.json').write_text(json.dumps(out,indent=2)+'\n')
 def test(label,base,prefix,rest,reference=False):
  frame=prepare(base);engine=Engine(Audit());v0,s0=old.evaluate(prefix,rest,frame,engine.positive)
  row=dict(label=label,n=len(rest),baseline_stats=s0,variants={})
  vv=prefix[:,None,:]+rest[:,None,None]*base[None,:,:]
  refs=np.array([engine.positive(v,6,1e-11,128)[0] for v in vv]) if reference else None
  if refs is not None:row['baseline_reference_relative']=float(np.max(np.abs(v0/refs-1)))
  for normalized in (False,True):
   name='normalized' if normalized else 'raw';engine=Engine(Audit())
   v,s=evaluate(prefix,rest,frame,engine.positive,normalize_edges=normalized)
   result=dict(relative_difference=float(np.max(np.abs(v/v0-1))),stats=s,same_fallbacks=s['positive_quadrature_fallbacks']==s0['positive_quadrature_fallbacks'])
   if refs is not None:result['reference_relative']=float(np.max(np.abs(v/refs-1)))
   row['variants'][name]=result
   if result['relative_difference']>1e-9 or not result['same_fallbacks']:out['diagnosis_required']=True
  out['rows'].append(row);save()
 eye=np.eye(3,6);anchor=.2*np.ones(6)/6+.8*np.eye(6)[0];cases=[]
 for width in (1.,.1,.01,.001,.0001,.00001,.0000001,0.):
  b=np.ones((3,6))+width*eye;b/=b.sum(axis=1)[:,None];cases.append((['symmetric',width],b))
 for width in (.1,.01,.001,.0001,.00001,.0000001,0.):
  b=np.stack([anchor,anchor+width*(eye[1]-eye[0]),anchor+width*(eye[2]-eye[0])]);cases.append((['asymmetric',width],b))
 for label,b in cases:
  for perm in itertools.permutations(range(3)):test(label+[list(perm)],b[list(perm)],np.zeros((1,6)),np.ones(1),True)
 test('zero_remainder',eye,anchor[None,:],np.zeros(1),True)
 test('coincident_edge',np.stack([eye[0],eye[0],eye[1]]),np.zeros((1,6)),np.ones(1),True)
 fixture=HERE.parent/'directed_chebyshev/t6_fixture.json';rays=np.array(json.loads(fixture.read_text())['rays'])
 for n in (4,8):
  y,rest,w=outer_rule(n,Audit())
  for i,ray in enumerate(rays):test(['stored',i,n],ray[:,3:].T,y@ray[:,:3].T,rest)
 ray=rays[2];y,rest,w=outer_rule(8,Audit());prefix=y@ray[:,:3].T
 for perm in itertools.permutations(range(3)):test(['stored_permutation',2,8,list(perm)],ray[:,3:].T[list(perm)],prefix,rest)
 test('sensitive447',ray[:,3:].T,prefix[447:448],rest[447:448],True)
 out['max_relative']={name:max(r['variants'][name]['relative_difference'] for r in out['rows']) for name in ('raw','normalized')}
 out['changed_fallback_rows']=[dict(label=r['label'],variant=name) for r in out['rows'] for name,v in r['variants'].items() if not v['same_fallbacks']]
 save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','source_hashes')},indent=2))
if __name__=='__main__':main()
