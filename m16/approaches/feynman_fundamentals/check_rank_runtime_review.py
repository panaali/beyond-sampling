"""Independent contraction of all sixteen native-census chunks with cold setup once."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from collections import Counter
from pathlib import Path
import json,hashlib,time
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census');S=json.loads((P/'summary.json').read_text());E=json.loads((P.parent/'rank_cost_review/environment.json').read_text());out=[]
for p,h in S['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
for mode,kind,N,cells,legacy in [('exact','exact',35098,8455,35650854),('recipe108','recipe',28000,6182,36398526)]:
 batches=[json.loads((P/f'{mode}_{i}.json').read_text()) for i in range(8)];cold=batches[0]['cold'];counts=Counter();ranks=Counter();iteration=q3=0;maximum=0
 for i,b in enumerate(batches):
  assert b['first']==12*i and b['last']==12*(i+1) and b['batch']==i and b['cold']==cold
  for p,h in b['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
  old=json.loads((P.parent/('plackett_cost_review' if mode=='exact' else 'recipe_cost_review')/f'geometry_{i}.json').read_text());assert all(b[k]==old[k] for k in ('legacy_geometry','stats','mass_callbacks'))
 for e in [b['native'] for b in batches]+[cold]:
  assert e['overflow']==0 and not any(e['nested'][i] for i in (3,7,11));assert e['gesdd'][0]==e['gesdd'][1]==e['gesdd'][8] and not e['gesdd'][9] and not e['gesdd'][10]
  for r in e['kernels']:
   assert r['failures']==0;counts[(r['kernel'],r['m'],r['n'],r['query'])]+=r['calls']
   if r['kernel']=='dgesdd' and not r['query']:ranks[str((r['m'],r['n']))]+=r['calls']
  iteration+=e['nested'][8];q3+=e['nested'][10];maximum=max(maximum,e['nested'][9])
 target=S['modes'][mode];reported=Counter({(r['kernel'],r['m'],r['n'],r['query']):r['calls'] for r in target['rows']});assert counts==reported and dict(ranks)==E['census'][kind]['shapes']==target['rank_shapes'];kinds=Counter()
 for (name,m,n,query),count in counts.items():kinds[name+('_query' if query else '')]+=count
 assert dict(sorted(kinds.items()))==target['kernel_calls'];assert iteration==kinds['dlasq5']+kinds['dlasq6']==target['dqds_iterations'] and q3==kinds['dlasq3']==target['q3_calls'] and maximum==target['max_dqds_iterations_per_q3']==3
 assert sum(b['mass_callbacks'] for b in batches)==N and sum(b['stats']['cells'] for b in batches)==cells and sum(b['legacy_geometry'] for b in batches)==legacy
 assert cold['gesdd'][3]==target['cold_rank_calls']==768 and sum(ranks.values())==target['rank_calls'];assert kinds['dgesdd']==kinds['dgesdd_query']==sum(ranks.values())
 out.append(dict(mode=mode,ordered_mass_callbacks=N,cells=cells,rank_executions_including_cold_once=sum(ranks.values()),cold_rank_executions=768,q3_calls=q3,q5_calls=kinds['dlasq5'],q6_calls=kinds['dlasq6'],observed_iteration_increments=iteration,max_observed_iterations_per_q3=maximum,author_cpu=sum(b['cpu_seconds'] for b in batches)))
result=dict(chunks=16,modes=out,review_cpu=time.process_time()-start,scope='Independent saved-census contraction and prior geometry/stat/source checks; one cold setup per mode. Finite evaluated calls only, no universal convergence proof or numeric kernel-operation prices; parent/child work must not be double-counted.')
(H/'rank_runtime_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
