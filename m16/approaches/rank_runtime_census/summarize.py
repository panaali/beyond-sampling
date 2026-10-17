"""Contract finite native call ledgers; no guessed numeric-operation conversion."""
from collections import Counter
from pathlib import Path
import ast,hashlib,json
HERE=Path(__file__).resolve().parent
ENV=json.loads((HERE.parent/'rank_cost_review/environment.json').read_text())
outputs={}
for mode,original in [('exact','exact'),('recipe108','recipe')]:
 batches=[json.loads((HERE/f'{mode}_{i}.json').read_text()) for i in range(8)];calls=Counter();failures=Counter();rank=Counter();cold=batches[0]['cold']
 for i,b in enumerate(batches):
  assert b['batch']==i and b['first']==12*i and b['last']==12*(i+1)
  assert b['source_sha256']==hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest()
  assert b['cold']==cold
  for x in b['native']['kernels']:
   key=(x['kernel'],x['m'],x['n'],x['query']);calls[key]+=x['calls'];failures[key]+=x['failures']
   if x['kernel']=='dgesdd' and not x['query']:rank[str((x['m'],x['n']))]+=x['calls']
 geometry_rank_calls=sum(rank.values())
 # The earlier census includes one cold construction of the 96 parent cones.
 # Eight diagnostic processes repeat it, but a single estimator pays it once.
 for x in cold['kernels']:
  key=(x['kernel'],x['m'],x['n'],x['query']);calls[key]+=x['calls'];failures[key]+=x['failures']
  if x['kernel']=='dgesdd' and not x['query']:rank[str((x['m'],x['n']))]+=x['calls']
 events=[b['native'] for b in batches]+[cold]
 assert dict(rank)==ENV['census'][original]['shapes'],(mode,rank)
 assert sum(failures.values())==0
 kinds=Counter()
 for (name,m,n,query),v in calls.items():kinds[name+('_query' if query else '')]+=v
 q3iter=sum(e['nested'][8] for e in events)
 assert q3iter==kinds['dlasq5']+kinds['dlasq6']
 assert kinds['dgesdd']==kinds['dgesdd_query']==sum(rank.values())
 assert kinds['dlasq3']==sum(e['nested'][10] for e in events)
 for b in batches:assert b['native']['gesdd'][0]==b['native']['gesdd'][1]==b['native']['gesdd'][8] and b['native']['gesdd'][9]==b['native']['gesdd'][10]==0
 out=dict(mode=mode,geometry_rank_calls=geometry_rank_calls,cold_rank_calls=cold['gesdd'][3],rank_calls=sum(rank.values()),rank_shapes=dict(rank),kernel_calls=dict(sorted(kinds.items())),
  rows=[dict(kernel=k[0],m=k[1],n=k[2],query=k[3],calls=v,failures=failures[k]) for k,v in sorted(calls.items())],
  q1_calls=sum(e['nested'][0] for e in events),q2_calls=sum(e['nested'][4] for e in events),q3_calls=kinds['dlasq3'],dqds_iterations=q3iter,
  max_dqds_iterations_per_q3=max(e['nested'][9] for e in events),
  q1_failures=sum(e['nested'][3] for e in events),q2_failures=sum(e['nested'][7] for e in events),
  cold_setup_once=cold,mass_callbacks=sum(b['mass_callbacks'] for b in batches),cells=sum(b['stats']['cells'] for b in batches),legacy_geometry=sum(b['legacy_geometry'] for b in batches),
  cpu_seconds=sum(b['cpu_seconds'] for b in batches),max_rss_kib=max(b['max_rss_kib'] for b in batches),
  observed_only=True,numeric_operations_complete=False,no_probability_computation=True)
 assert out['mass_callbacks']==(35098 if mode=='exact' else 28000)
 assert out['cells']==(8455 if mode=='exact' else 6182)
 assert out['legacy_geometry']==(35650854 if mode=='exact' else 36398526)
 outputs[mode]=out
result=dict(modes=outputs,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'run.py',HERE/'spy.c',HERE/'observer.py',HERE/'spy_manifest.json']},
 cost_note='Nested kernel counts are not additive operation prices. Parent local work and descendant work must be separated before any envelope. Missing native helpers remain opaque; no whole-cost or universal convergence claim.')
(HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({m:{k:v for k,v in x.items() if k not in ('rows','rank_shapes','cold_setup_once')} for m,x in outputs.items()},indent=2))
