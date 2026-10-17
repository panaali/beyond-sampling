"""Full saved first64 k28 two-recipe pilot and exact budget audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import gzip,hashlib,json,time
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/review/margin_k28_native')
D=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery/frozen')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time();p=read(P/'pilot.json');s=read(P/'pilot_summary.json');c=read(P/'controls.json');recon=read(P/'reconstruction.json')
assert p['complete'] and s['complete'] and p['copies']==s['rows']==len(p['rows'])==64
assert p['driver_sha256']==sha(P/'pilot.py') and p['gate_sha256']==sha(P/'pilot_gate.json')
assert p['controls_sha256']==sha(P/'controls.json') and p['reconstruction_sha256']==sha(P/'reconstruction.json')
for path,h in read(P/'pilot_gate.json')['pins'].items():assert sha(Path(path))==h
for path,h in s['pins'].items():assert sha(Path(path))==h
assert sha(P/'pilot.jsonl.gz')==p['archive_sha256']
with gzip.open(P/'pilot.jsonl.gz','rt') as f:records=[json.loads(line) for line in f]
base={};pairs={}
for r in records:
 key=(r['copy'],r['recipe'])
 if r['stage']=='baseline':assert key not in base;base[key]=r['baseline']
 else:assert r['stage']=='paired_before_validation' and key not in pairs;pairs[key]=r
assert len(records)==254 and len(base)==126 and len(pairs)==128
truth=read(D/'rung_k28.json')['seeds']['0']['panel']['truths'];old=read(D/'margin_k28.json')['seeds']['0']['p20']
control=next(r for r in c['rows'] if (r['seed'],r['noise'],r['copy'])==(0,20,0))
counts={name:Counter() for name in ('r85b','r112')}
assert [r['copy'] for r in p['rows']]==list(range(64))
for j,r in enumerate(p['rows']):
 assert r['model_sha256']==recon['records'][0]['noises'][0]['hashes'][j] and r['truth']==truth[j] and r['reused_control']==(j==0)
 for name,v in r['recipes'].items():
  assert name in counts;raw=pairs[(j,name)];b,t=raw['baseline'],raw['traced']
  assert raw['model_sha256']==r['model_sha256'] and raw['reused_control']==(j==0)
  if j:assert base[(j,name)]==b
  else:assert b==control['recipes'][name]['baseline'] and t==control['recipes'][name]['traced']
  for k in ('value','hex','legacy','stats','info','recipe'):assert b[k]==t[k]
  assert v['value']==b['value'] and v['hex']==b['hex']==float(v['value']).hex()
  assert v['value']-r['truth']==v['old_error']==old[name]['errs'][j]
  assert v['legacy']==b['legacy'] and v['cells']==b['stats']['cells']
  l=t['ledger'];cnt=Counter()
  for site in l['sites'].values():cnt.update(site)
  assert cnt==Counter(l['totals']);integer=sum(x for k,x in cnt.items() if k.startswith('integer_control'));numeric=sum(cnt.values())-integer
  assert (numeric,integer)==(v['numeric'],v['integer'])==(l['named_numeric_total'],l['integer_control_total'])
  assert not l['complete_implementation_cost'];counts[name].update(cnt)
assert len({r['model_sha256'] for r in p['rows']})==64
ratios={}
mean=lambda x:sum(x,F())/64
for name in counts:
 assert dict(counts[name])==s['ledger_totals'][name];ratios[name]={}
 for conv,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
  g=s['groups'][name][conv];ps=[F(r['truth']) for r in p['rows']];err=[(F(r['recipes'][name]['value'])-v)**2 for r,v in zip(p['rows'],ps)]
  costs=[r['recipes'][name][key] for r in p['rows']];ns=[v//sampler for v in costs];pq=[v*(1-v) for v in ps];den=[v/n for v,n in zip(pq,ns)]
  assert float(mean(err))==g['mse'] and float(mean(den))==g['mean_sampling_variance_integer'] and float(mean(err)/mean(den))==g['ratio_integer_budget']
  assert float(mean(err)/mean(v*sampler/o for v,o in zip(pq,costs)))==g['ratio_continuous_budget']
  assert float(mean(err)/(mean(pq)*sampler/mean(map(F,costs))))==g['ratio_mean_budget']
  individuals=[e/v for e,v in zip(err,den)]
  assert sum(z>1 for z in individuals)==g['individual_losses']==0 and float(max(individuals))==g['max_individual_ratio']
  ix=np.random.default_rng(0).integers(0,64,(1000,64));boot=np.array(list(map(float,err)))[ix].mean(1)/np.array(list(map(float,den)))[ix].mean(1)
  assert np.array_equal(np.percentile(boot,[2.5,97.5]),g['bootstrap95_integer']);ratios[name][conv]=g['ratio_integer_budget']
assert s['pilot_cpu']==p['cpu'] and s['projected_80_shard_cpu']==80*p['cpu']
out=dict(complete=True,models=64,raw_records=254,paired_records=128,pilot_sha256=sha(P/'pilot.json'),summary_sha256=sha(P/'pilot_summary.json'),
 numeric={name:sum(r['recipes'][name]['numeric'] for r in p['rows']) for name in counts},ratios=ratios,
 research_cpu=p['cpu'],projected_80_shard_cpu=s['projected_80_shard_cpu'],review_cpu=time.process_time()-start,new_evaluator_calls=0,
 scope='Full saved site/pin/hash/error parity, single reuse and exact own-budget/bootstrap summaries pass. Current canonical first64 only; historical-byte identity/noisy counters absent, conditional truth and incomplete native/runtime scope remain. Expansion requires separately frozen source/resource gate.')
Path(__file__).with_name('k28_native_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
