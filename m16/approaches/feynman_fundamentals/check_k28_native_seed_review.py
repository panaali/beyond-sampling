"""Independent complete-seed k28 saved audit, no evaluator calls."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,gzip,hashlib,time,sys
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/review/margin_k28_native')
D=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery/frozen')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ledger(l):
 c=Counter()
 for vals in l['sites'].values():
  assert all(type(v) is int and v>=0 for v in vals.values());c.update(vals)
 assert dict(c)==l['totals'] and not l['complete_implementation_cost']
 integer=sum(v for k,v in c.items() if k.startswith('integer_control'));numeric=sum(c.values())-integer
 assert (numeric,integer)==(l['named_numeric_total'],l['integer_control_total']);return c,numeric,integer
start=time.process_time();seed=int(sys.argv[1]);assert seed in range(5)
target=Path(__file__).with_name(f'k28_native_seed{seed}_review_results.json');assert not target.exists()
for gate in ('pilot_gate.json','continuation_gate.json'):
 for p,h in read(P/gate)['pins'].items():assert sha(p)==h
controls=read(P/'controls.json');recon=read(P/'reconstruction.json');old=read(D/'margin_k28.json')['seeds'][str(seed)];truth=read(D/'rung_k28.json')['seeds'][str(seed)]['panel']['truths']
cmap={(r['seed'],r['noise'],r['copy']):r for r in controls['rows']};identity=recon['records'][seed];assert identity['seed']==seed
rows={n:[] for n in ('r85b','r112')};totals={n:Counter() for n in rows};files={};hashes=set();reuses=[];cpu=0.;maxcpu=0.;paired_count=baseline_count=0
for noise in range(20,24):
 ident=identity['noises'][noise-20];assert ident['noise']==noise
 for block in range(4):
  pilot=(seed,noise,block)==(0,20,0);path=P/('pilot.json' if pilot else f'panel/s{seed}_n{noise}_b{block}.json');d=read(path)
  assert d['complete'] and 'failure' not in d and (d['seed'],d['noise'],d['block'])==(seed,noise,block)
  assert d['driver_sha256']==sha(P/('pilot.py' if pilot else 'shard.py')) and d['gate_sha256']==sha(P/('pilot_gate.json' if pilot else 'continuation_gate.json'))
  assert d['controls_sha256']==sha(P/'controls.json') and d['reconstruction_sha256']==sha(P/'reconstruction.json') and not d['historical_identity_verified']
  assert [r['copy'] for r in d['rows']]==list(range(64*block,64*(block+1)))
  archive=path.with_suffix('.jsonl.gz');assert sha(archive)==d['archive_sha256'];files[str(path.relative_to(P))]=sha(path);files[str(archive.relative_to(P))]=sha(archive)
  baselines={};pairs={}
  with gzip.open(archive,'rt') as stream:
   for line in stream:
    raw=json.loads(line);key=(raw['copy'],raw['recipe'])
    if raw['stage']=='baseline':assert key not in baselines;baselines[key]=raw['baseline']
    else:assert raw['stage']=='paired_before_validation' and key not in pairs;pairs[key]=raw
  assert set(pairs)=={(j,n) for j in range(64*block,64*(block+1)) for n in rows};paired_count+=len(pairs);baseline_count+=len(baselines)
  for row in d['rows']:
   j=row['copy'];digest=row['model_sha256'];assert digest==ident['hashes'][j] and digest not in hashes;hashes.add(digest)
   p=truth[(noise-20)*256+j];assert row['truth']==p;control=cmap.get((seed,noise,j));reuse=control is not None;assert row['reused_control']==reuse
   if reuse:reuses.append([seed,noise,j]);assert control['model_sha256']==digest and control['truth']==p
   assert set(row['recipes'])==set(rows)
   for name,v in row['recipes'].items():
    raw=pairs[j,name];b,t=raw['baseline'],raw['traced'];assert raw['model_sha256']==digest and raw['reused_control']==reuse
    if reuse:assert (j,name) not in baselines and b==control['recipes'][name]['baseline'] and t==control['recipes'][name]['traced']
    else:assert baselines.pop((j,name))==b
    for key in ('hex','value','legacy','stats','info','recipe'):assert b[key]==t[key]
    assert b['hex']==v['hex']==float(v['value']).hex() and v['value']==b['value'] and v['value']-p==v['old_error']==old['p20'][name]['errs'][(noise-20)*256+j]
    assert v['legacy']==b['legacy'] and v['cells']==b['stats']['cells']
    count,numeric,integer=ledger(t['ledger']);assert (numeric,integer)==(v['numeric'],v['integer']);totals[name].update(count)
    rows[name].append(dict(v,seed=seed,noise=noise,copy=j,truth=p,model_sha256=digest))
  assert not baselines;cpu+=d['cpu'];maxcpu=max(maxcpu,d['cpu'])
assert len(hashes)==1024 and paired_count==2048 and baseline_count==2044 and reuses==[[seed,20,0],[seed,23,255]]
def metrics(rr):
 N=len(rr);mean=lambda xs:sum(xs,F())/N;out={}
 for conv,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
  ps=[F(r['truth']) for r in rr];es=[(F(r['value'])-p)**2 for r,p in zip(rr,ps)];costs=[r[key] for r in rr];ns=[c//sampler for c in costs];pq=[p*(1-p) for p in ps];vs=[q/n for q,n in zip(pq,ns)];individual=[e/v for e,v in zip(es,vs)];worst=individual.index(max(individual))
  ix=np.random.default_rng(0).integers(0,N,(1000,N));boot=np.array(list(map(float,es)))[ix].mean(1)/np.array(list(map(float,vs)))[ix].mean(1)
  out[conv]=dict(mse=float(mean(es)),mean_sampling_variance_integer=float(mean(vs)),ratio_integer_budget=float(mean(es)/mean(vs)),ratio_continuous_budget=float(mean(es)/mean(q*sampler/c for q,c in zip(pq,costs))),ratio_mean_budget=float(mean(es)/(mean(pq)*sampler/mean(map(F,costs)))),mean_ops=float(mean(map(F,costs))),min_samples=min(ns),max_samples=max(ns),bootstrap95_integer=np.percentile(boot,[2.5,97.5]).tolist(),individual_losses=sum(x>1 for x in individual),max_individual_ratio=float(max(individual)),worst_copy={k:rr[worst][k] for k in ('seed','noise','copy')})
 return out
means={}
for name,rr in rows.items():
 assert len(rr)==1024;means[name]={}
 for key,savedkey in [('legacy','ops'),('cells','cells')]:
  v=F(sum(r[key] for r in rr),1024);assert v==F(old['p20'][name][savedkey]);means[name][key]=str(v)
metrics_all={name:metrics(rr) for name,rr in rows.items()};noise_metrics={str(n):{name:metrics([r for r in rr if r['noise']==n]) for name,rr in rows.items()} for n in range(20,24)}
individual=[dict(recipe=name,seed=seed,noise=r['noise'],copy=r['copy'],model_sha256=r['model_sha256'],ratios={conv:float((F(r['value'])-F(r['truth']))**2*(r[key]//sampler)/(F(r['truth'])*(1-F(r['truth'])))) for conv,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]}) for name,rr in rows.items() for r in rr]
out=dict(complete=True,seed=seed,models=1024,paired_records=paired_count,baseline_records=baseline_count,files=files,reuses=reuses,metrics=metrics_all,noise_metrics=noise_metrics,individual=individual,totals={n:dict(c) for n,c in totals.items()},numeric={n:sum(r['numeric'] for r in rr) for n,rr in rows.items()},integer={n:sum(r['integer'] for r in rr) for n,rr in rows.items()},original_mean_counter_parity=means,research_cpu=cpu,max_shard_cpu=maxcpu,review_cpu=time.process_time()-start,new_evaluator_calls=0,scope='All saved sites, paired/baseline records, current model hashes and old subtraction errors verified. Historical checkpoint byte identity remains unproved; no missing historical per-copy costs invented. Source numeric111/historical72 and conditional bootstrap retained, not full native/runtime.')
target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('complete','seed','numeric','integer','review_cpu','metrics')}))
