from pathlib import Path
from collections import Counter
import ast,copy,hashlib,itertools,json,math,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
import numpy as np
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/e46_e48_review/seed2_reference');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((P/'capture_protocol.json').read_text());r=json.loads((P/'capture.json').read_text());assert r['complete'] and r['protocol_sha256']==sha(P/'capture_protocol.json')
for file,h in p['pins'].items():assert sha(Path(file))==h,file
assert r['checkpoint_sha256']==sha(Path(p['checkpoint']))
# Independent read-only checkpoint decode, no network construction/forward.
import torch
torch.set_num_threads(1);weights=torch.load(p['checkpoint'],map_location='cpu',weights_only=True)
keys=['rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'];assert set(weights)==set(keys)
shapes=[(8,1),(8,8),(4,8)]
for key,shape,saved in zip(keys,shapes,r['model_hex'],strict=True):
 t=weights[key];assert t.dtype==torch.float32 and tuple(t.shape)==shape and bool(torch.isfinite(t).all())
 data=t.tolist();want=[float(x[0]).hex() for x in data] if key==keys[0] else [[float(x).hex() for x in row] for row in data]
 assert want==saved
# Source agreement: exact branch only, and separately bound globals.
old=ast.parse(Path(p['old_engine']).read_text());new=ast.parse(Path(p['current_engine']).read_text());dump=lambda x:ast.dump(x,include_attributes=False)
def fun(t):return {x.name:x for x in t.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
a,b=fun(old),fun(new)
for name,x in a.items():
 if name=='orthant':continue
 y=copy.deepcopy(b[name])
 if name=='_margin_split':
  diag=[n for n in y.body if isinstance(n,ast.If) and ast.unparse(n.test)=='DIAG[0] is not None and (not exact)'];assert len(diag)==1
  assert ast.unparse(diag[0])=='if DIAG[0] is not None and (not exact):\n    for k in trig:\n        DIAG[0](c, A, g, k, t, model, truth)'
  y.body.remove(diag[0])
 assert dump(x)==dump(y),name
for name in ('TOL','MODE','EXACT_STEPS','STATS','_HALVES'):
 def assign(t):return next(n for n in t.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
 assert dump(assign(old))==dump(assign(new))
progress=[json.loads(x) for x in (P/'capture_progress.jsonl').read_text().splitlines()]
assert len(progress)==195 and progress[0]['stage']=='preflight_attempt' and progress[1]['stage']=='capture_start' and progress[-1]['stage']=='capture_complete'
assert [x['row'] for x in progress if x['stage']=='parent_complete']==r['parents']
assert [x['index'] for x in progress if x['stage']=='parent_attempt']==list(range(96))
truths=[perm[1] for sign in (1,-1) for perm in itertools.permutations(range(4)) for side in (1,-1)]
assert len(r['parents'])==r['parent_archive_count']==96
stats=Counter();N=0;payload=hashlib.sha256();names=[]
for i,row in enumerate(r['parents']):
 assert row['index']==i and row['truth']==truths[i];file=P/'parents'/row['archive'];assert file.name==f'parent_{i:02d}.npz' and sha(file)==row['sha256'];names.append(file.name)
 with np.load(file,allow_pickle=False) as z:
  assert z.files==['rays'];ray=z['rays'];assert ray.dtype==np.dtype('float64') and ray.shape==(row['N'],4,4) and ray.flags.c_contiguous and np.isfinite(ray).all()
  payload.update(ray.tobytes(order='C'));assert row['copied_float64_elements']==row['archive_stacking_float64_elements']==ray.size and row['ray_copy_bytes']==ray.nbytes
 assert row['stats_delta']['simplices']==row['N'];stats.update(row['stats_delta']);N+=row['N']
assert sorted(x.name for x in (P/'parents').iterdir())==names
assert dict(stats)==r['stats'] and N==r['N']==37739 and stats['cells']==p['expected_leaves']==9470 and stats['resolved']==9374
assert r['ray_copy_bytes']==128*N and r['auxiliary_proxy_instances']==[595]*96
assert r['all_proxy_instances_sum']==r['main_geometry_proxy']+96*595==38450554
assert r['guard_counts']['scalar_equalities']//4==r['guard_counts']['boolean_reductions']//3==25539
assert r['guard_counts']['scalar_equalities']==102156 and r['guard_counts']['boolean_reductions']==76617 and set(r['guard_counts'])=={'scalar_equalities','boolean_reductions'}
assert r['mass_evaluations']==r['random_inputs']==r['neural_forward_input_evaluations']==0 and r['ignored_zero_callback_accuracy']==1
assert sum(x['cpu_including_archive'] for x in r['parents'])<=r['cpu']<=r['process_cpu']
v=json.loads((P/'serialization_check.json').read_text());assert v['complete'] and v['N']==N and v['ordered_float64_payload_sha256']==payload.hexdigest()
result=dict(complete=True,parents=96,rays=N,stats=dict(stats),checkpoint_sha256=r['checkpoint_sha256'],capture_sha256=sha(P/'capture.json'),ordered_payload_sha256=payload.hexdigest(),guard_comparisons=102156,guard_boolean_reductions=76617,main_proxy=r['main_geometry_proxy'],auxiliary_proxy=96*595,total_proxy=r['all_proxy_instances_sum'],author_cpu=r['cpu'],review_cpu=time.process_time()-start,review_checkpoint_loads=1,scope='Independent read-only checkpoint decoding, exact-mode source/globals compatibility, all96 archive bytes/order/truth labels/progress/totals/guard accounting. Zero-callback score is ignored. No geometry/forward/Gaussian mass replay; existing approximate clipping/rank and historical checkpoint identity remain unproved. Inherited proxy is not complete native cost.')
(O/'seed2_capture_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
