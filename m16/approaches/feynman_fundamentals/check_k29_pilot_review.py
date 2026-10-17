import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import json,gzip,hashlib,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_910/review/margin_k29_native');D=Path('/workspace/work/recover-small_h2_56/m16/approaches/training_axis_ladder');start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ledger(l):
 c=Counter()
 for v in l['sites'].values():c.update(v)
 assert c==Counter(l['totals']);ints=sum(v for k,v in c.items() if k.startswith('integer_control'));num=sum(c.values())-ints;assert (num,ints)==(l['named_numeric_total'],l['integer_control_total']) and not l['complete_implementation_cost'];return num,ints
prov=read(P/'provenance.json');assert prov['complete'] and prov['rows']==5120 and len(prov['blocks'])==20
for path,h in prov['pins'].items():assert sha(Path(path))==h
pilot=read(P/'pilot.json');s=read(P/'pilot_summary.json');assert pilot['complete'] and pilot['copies']==64 and pilot['provenance_sha256']==sha(P/'provenance.json') and pilot['archive_sha256']==sha(P/'pilot.jsonl.gz') and s['pilot_sha256']==sha(P/'pilot.json');assert [r['copy'] for r in pilot['rows']]==list(range(64))
controls={}
for seed in (0,4):
 c=read(P/f'control_s{seed}.json');assert c['complete'] and sha(P/f'control_s{seed}.json')==pilot['controls'][str(seed)];controls[seed]=c
 for name,v in c['recipes'].items():
  for key in ('hex','legacy','stats','info','recipe'):assert v['baseline'][key]==v['traced'][key]
  ledger(v['traced']['ledger'])
old=read(D/'block_s0_n20.json');totals=Counter()
with gzip.open(P/'pilot.jsonl.gz','rt') as f:raw=[json.loads(line) for line in f]
assert len(raw)==64
for j,(r,a) in enumerate(zip(pilot['rows'],raw)):
 saved=dict(zip(old['schema'],old['rows'][j]));assert a['copy']==j and r['model_sha256']==a['model_sha256']==saved['model_sha256'] and r['truth']==saved['truth'] and r['reused_control']==(j==0)
 for name,v in a['recipes'].items():
  b,t=v['baseline'],v['traced'];q=r['recipes'][name]
  for key in ('hex','legacy','stats','info','recipe'):assert b[key]==t[key]
  num,ints=ledger(t['ledger']);assert (num,ints)==(q['numeric'],q['integer']) and q['value']==b['value'] and q['hex']==float(q['value']).hex()==b['hex'] and q['legacy']==b['legacy'] and q['cells']==b['stats']['cells'];totals[name]+=num
  if j==0:assert v==controls[0]['recipes'][name]
ratios={}
for name in ('r85b','r112'):
 for conv,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
  g=s['groups'][name][conv];ps=[F(r['truth']) for r in pilot['rows']];es=[(F(r['recipes'][name]['value'])-p)**2 for r,p in zip(pilot['rows'],ps)];ops=[r['recipes'][name][key] for r in pilot['rows']];ns=[o//sampler for o in ops];vs=[p*(1-p) for p in ps];den=[v/n for v,n in zip(vs,ns)];mean=lambda x:sum(x,F())/64;mse=mean(es);assert float(mse)==g['mse'] and float(mean(den))==g['mean_sampling_variance_integer'] and float(mse/mean(den))==g['ratio_integer_budget'];assert float(mse/mean(v*sampler/o for v,o in zip(vs,ops)))==g['ratio_continuous_budget'];assert float(mse/(mean(vs)*sampler/mean(map(F,ops))))==g['ratio_mean_budget'];assert sum(e>d for e,d in zip(es,den))==g['individual_losses']==0;assert min(ns)==g['min_samples'] and max(ns)==g['max_samples'] and float(mean(map(F,ops)))==g['mean_ops']
  ix=np.random.default_rng(0).integers(0,64,(1000,64));boot=np.array(list(map(float,es)))[ix].mean(1)/np.array(list(map(float,den)))[ix].mean(1);assert np.array_equal(np.percentile(boot,[2.5,97.5]),g['bootstrap95_integer']);ratios[name,conv]=g['ratio_integer_budget']
out=dict(models=64,recipes=2,control_models=2,native_totals=dict(totals),ratios={str(k):v for k,v in ratios.items()},review_cpu=time.process_time()-start,scope='Saved provenance hashes, all128 native ledgers/parity records, control0reuse once, exact-rational budget contractions and frozen bootstrap reproduced independently. Existing reviewed numerical truth reused, no estimator/model/truth evaluations; first64cost gate only, integer/control/memory/library/setup exclusions retained.')
Path(__file__).with_name('k29_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
