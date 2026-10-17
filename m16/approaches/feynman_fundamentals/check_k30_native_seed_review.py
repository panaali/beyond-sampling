"""Bounded per-seed saved audit; can precede final all-panel summary."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import json,gzip,hashlib,time,sys
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/review/margin_k30_native');D=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery');R=Path('/workspace/work/recover-small_h2_910/review/recipe85b_native_cost');start=time.process_time();seed=int(sys.argv[1]);assert seed in range(1,5)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ledger(l):
 c=Counter()
 for v in l['sites'].values():c.update(v)
 assert c==Counter(l['totals']) and not l['complete_implementation_cost'];i=sum(v for k,v in c.items() if k.startswith('integer_control'));n=sum(c.values())-i
 assert (n,i)==(l['named_numeric_total'],l['integer_control_total']);return n,i,c
rows=[];pins={};totals=Counter();cpu=0.;maxcpu=0.;reuses=[]
for noise in range(20,24):
 prior=read(D/f'k30_s{seed}_noise{noise}.json')
 for block in range(4):
  path=P/('pilot.json' if (seed,noise,block)==(1,20,0) else f'panel/s{seed}_n{noise}_b{block}.json');d=read(path);assert d['complete'] and 'failure' not in d
  assert (d['seed'],d['noise'],d['block'])==(seed,noise,block) and [r['copy'] for r in d['rows']]==list(range(64*block,64*(block+1)))
  assert d['provenance_sha256']==sha(P/'provenance.json') and d['driver_sha256']==sha(P/('pilot.py' if path.name=='pilot.json' else 'shard.py'))
  if path.name!='pilot.json':assert d['pilot_gate_sha256']==sha(P/'pilot_review.json')
  archive=path.with_suffix('.jsonl.gz');assert sha(archive)==d['archive_sha256'];pins[str(path.relative_to(P))]=sha(path)
  with gzip.open(archive,'rt') as f:
   for row in d['rows']:
    raw=json.loads(next(f));j=row['copy'];saved=prior['rows'][j]
    assert raw['copy']==j and raw['model_sha256']==row['model_sha256']==saved['model_hash'] and row['truth']==saved['saved_truth']
    b,t=raw['baseline'],raw['traced']
    for key in ('hex','legacy','stats','info'):assert b[key]==t[key]
    num,integer,c=ledger(t['ledger']);totals.update(c)
    assert (num,integer)==(row['numeric'],row['integer']) and row['hex']==float(row['value']).hex()==b['hex']
    assert row['value']==b['value']==saved['recipes']['r85b']['value'] and row['legacy']==b['legacy']==saved['recipes']['r85b']['native_ops'] and row['cells']==b['stats']['cells']==saved['recipes']['r85b']['cells']
    reused=(seed,noise,j) in ((1,20,0),(4,23,255));assert reused==row['reused_control']
    if reused:
     control=read(R/f'trained_seed{seed}.json');assert b==control['baseline'] and t==control['traced'];reuses.append([seed,noise,j])
    rows.append(dict(row,seed=seed,noise=noise))
   assert next(f,None) is None
  cpu+=d['cpu'];maxcpu=max(maxcpu,d['cpu'])
assert len(rows)==len({r['model_sha256'] for r in rows})==1024

def metrics(rr):
 out={};N=len(rr);mean=lambda x:sum(x,F())/N
 for convention,sampler,cost in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
  ps=[F(r['truth']) for r in rr];es=[(F(r['value'])-p)**2 for r,p in zip(rr,ps)];ops=[r[cost] for r in rr];ns=[o//sampler for o in ops];pq=[p*(1-p) for p in ps];v=[q/n for q,n in zip(pq,ns)];individual=[e/u for e,u in zip(es,v)];ix=np.random.default_rng(0).integers(0,N,(1000,N));boot=np.array(list(map(float,es)))[ix].mean(1)/np.array(list(map(float,v)))[ix].mean(1);worst=individual.index(max(individual))
  out[convention]=dict(mse=float(mean(es)),mean_sampling_variance_integer=float(mean(v)),ratio_integer_budget=float(mean(es)/mean(v)),ratio_continuous_budget=float(mean(es)/mean(q*sampler/o for q,o in zip(pq,ops))),ratio_mean_budget=float(mean(es)/(mean(pq)*sampler/mean(map(F,ops)))),mean_ops=float(mean(map(F,ops))),min_samples=min(ns),max_samples=max(ns),bootstrap95_integer=np.percentile(boot,[2.5,97.5]).tolist(),individual_losses=sum(x>1 for x in individual),max_individual_ratio=float(max(individual)),worst_copy={k:rr[worst][k] for k in ('seed','noise','copy')})
 return out
summary=metrics(rows);noisegroups={str(n):metrics([r for r in rows if r['noise']==n]) for n in range(20,24)}
individual=[dict(seed=seed,noise=r['noise'],copy=r['copy'],model_sha256=r['model_sha256'],ratios={name:float((F(r['value'])-F(r['truth']))**2*(r[key]//sampler)/(F(r['truth'])*(1-F(r['truth'])))) for name,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]}) for r in rows]
out=dict(complete=True,seed=seed,rows=1024,metrics=summary,noise_metrics=noisegroups,individual=individual,files=pins,totals=dict(totals),numeric=sum(r['numeric'] for r in rows),integer=sum(r['integer'] for r in rows),legacy=sum(r['legacy'] for r in rows),reuses=reuses,research_cpu=cpu,max_shard_cpu=maxcpu,review_cpu=time.process_time()-start,scope='Every saved native site/parity/source/model/truth record and exact own-budget metrics checked. Numerical truths inherited; bootstrap conditional, source111/historical72 separate; no native runtime or per-copy universal guarantee.')
Path(__file__).with_name(f'k30_native_seed{seed}_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('complete','seed','numeric','integer','review_cpu','metrics')}))
