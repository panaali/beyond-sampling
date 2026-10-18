"""Complete T184 (H,zoo) saved group audit, across preserved and recovered shards."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,gzip,hashlib,time,sys,resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_78/review/recipe112_core_native');OUT=Path(__file__).resolve().parent
H,Z=map(int,sys.argv[1:]);assert H in (4,5,6) and Z in range(5)
target=OUT/f'core_native_H{H}_z{Z}_review_results.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rec=read(P/'recovery_gate.json');pilots=read(P/'pilots.json');failrows=read(P/'recovery_evaluations.json')['rows'];manifest=read(P/'manifest.json')
for p,h in rec['pins'].items():assert sha(p)==h,p
for p,h in manifest['pins'].items():assert sha(p)==h,p
files=[];allrows=[];hashes=set();counter=Counter();topology=[];newcalls=0;reuses=Counter();research_cpu=0
sampler=4*H*H+12*H-1
for n in range(80,84):
 stem=f'H{H}_z{Z}_n{n}';isold=stem in rec['completed_shards'];assert isold or stem in rec['remaining_shards']
 path=P/('panel' if isold else 'panel_v2')/(stem+'.json');d=read(path);archive=path.with_suffix('.jsonl.gz');original=P/'frozen/core'/('core_'+stem+'.json');old=read(original)
 assert d['complete'] and (d['H'],d['zoo'],d['noise'])==(H,Z,n) and d['sampler_numeric']==sampler
 assert d['gate_sha256']==sha(P/('panel_gate.json' if isold else 'recovery_gate.json')) and sha(archive)==d['archive_sha256'] and sha(original)==d['prior_shard_sha256']
 assert d['checkpoint_sha256']==old['checkpoint_sha256']
 assert d['source_sha256']==sha(Path(manifest['source_root'])/'dependencies/understand_m43/est75.py') and d['trace_sha256']==sha(Path(manifest['native_root'])/'trace.py')
 assert [r['copy'] for r in d['rows']]==list(range(64)) and len(old['rows'])==64
 events={}
 with gzip.open(archive,'rt') as f:
  for line in f:
   e=json.loads(line)
   if e['event']=='evaluations':assert e['copy'] not in events;events[e['copy']]=e
 assert sorted(events)==list(range(64));pr=fr=0
 for i,(r,s) in enumerate(zip(d['rows'],old['rows'],strict=True)):
  e=events[i];b,t=e['baseline'],e['traced']
  for key in ('value','hex','legacy','stats','info'):assert b[key]==t[key],(stem,i,key)
  assert r['parity'] and r['stage']=='complete' and r['hex']==b['hex']==float(b['value']).hex()==float(r['estimate']).hex()==float(s['saved_estimate']).hex()
  assert r['legacy']==b['legacy']==s['saved_ops'] and r['truth']==s['truth'] and r['model_sha256']==s['model_sha256']
  assert r['model_sha256'] not in hashes;hashes.add(r['model_sha256'])
  counts=Counter()
  for site in t['ledger']['sites'].values():
   assert all(type(x)is int and x>=0 for x in site.values());counts.update(site)
  assert dict(counts)==t['ledger']['totals']==r['totals'] and not t['ledger']['complete_implementation_cost']
  integer=sum(v for k,v in counts.items() if k.startswith('integer_control'));numeric=sum(counts.values())-integer
  assert integer==r['integer_control']==t['ledger']['integer_control_total'] and numeric==r['named_numeric']==t['ledger']['named_numeric_total'];counter.update(counts)
  startup=Counter()
  for site in t['startup']['sites'].values():startup.update(site)
  assert dict(startup)==t['startup']['totals'] and sum(startup.values())==0
  pilot=Z==0 and n==80 and i==0;failed=stem==rec['accepted_failure'] and i in (0,1)
  if pilot:
   p=next(p for p in pilots['rows'] if p['H']==H);assert b==p['baseline'] and t==p['traced'];pr+=1;reuses['pilot']+=1
  if failed:assert b==failrows[i]['baseline'] and t==failrows[i]['traced'];fr+=1;reuses['failed_case']+=1
  if isold:assert r['reused_pilot']==pilot
  else:
   assert r['reused_from']==('pilot' if pilot else 'failed_v1_saved_evaluations' if failed else None)
   assert r['native_stats']==b['stats'] and r['saved_scalar_stats']==s['stats'] and r['saved_scalar_cells']==s['saved_cells']
   assert r['scalar_native_cells_equal']==(b['stats']['cells']==s['saved_cells'])
  assert r['cells']==b['stats']['cells']
  if b['stats']!=s['stats']:topology.append(dict(noise=n,copy=i,native=b['stats'],scalar=s['stats']))
  p=F(r['truth']);err=(F(r['estimate'])-p)**2;v=p*(1-p)/(numeric//sampler)
  allrows.append(dict(noise=n,copy=i,model_sha256=r['model_sha256'],numeric=numeric,integer=integer,error_squared=err,variance=v,individual_ratio=err/v))
 assert d['pilot_reuses']==pr and d.get('failed_attempt_reuses',0)==fr and d['source_calls_started']==d['source_calls_completed']==2*(64-pr-fr)
 newcalls+=d['source_calls_completed'];research_cpu+=d['cpu_seconds'];files.append(dict(path=str(path),sha256=sha(path),archive_sha256=sha(archive)))
assert len(hashes)==len(allrows)==256
out={'complete':True,'H':H,'zoo':Z,'models':256,'files':files,'numeric':sum(r['numeric'] for r in allrows),'integer':sum(r['integer'] for r in allrows),'categories':dict(counter),'sampler_numeric':sampler,'ratio_per_model_floor':float(sum(r['error_squared'] for r in allrows)/sum(r['variance'] for r in allrows)),'individual_losses':sum(r['individual_ratio']>1 for r in allrows),'individual_ties':sum(r['individual_ratio']==1 for r in allrows),'max_individual_ratio':float(max(r['individual_ratio'] for r in allrows)),'topology_differences':topology,'new_calls':newcalls,'reuses':dict(reuses),'research_cpu_completed_shards':research_cpu,'review_cpu':time.process_time()-start,'scope':'Complete saved sites/current native parity and original output/legacy/model binding. Scalar/native topology diagnostics retained. No estimator/model/truth calls. Declared named numeric allowance, not full native/runtime cost. Failed research reservation reconciled only at global stage.'}
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('categories','files','topology_differences')}));print('topology_differences',len(topology))
