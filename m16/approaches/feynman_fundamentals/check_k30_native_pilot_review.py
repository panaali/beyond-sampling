"""Independent saved-only first64 missing-k30 native ledger audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import json,gzip,hashlib,time
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/review/margin_k30_native')
D=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery')
R=Path('/workspace/work/recover-small_h2_910/review/recipe85b_native_cost')
start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ledger(l):
 c=Counter()
 for v in l['sites'].values():c.update(v)
 assert c==Counter(l['totals']);ints=sum(v for k,v in c.items() if k.startswith('integer_control'));num=sum(c.values())-ints
 assert (num,ints)==(l['named_numeric_total'],l['integer_control_total']) and not l['complete_implementation_cost'];return num,ints
prov=read(P/'provenance.json');assert prov['complete'] and prov['rows']==4096 and len(prov['blocks'])==16
for path,h in prov['pins'].items():assert sha(Path(path))==h
pilot=read(P/'pilot.json');s=read(P/'pilot_summary.json')
assert pilot['complete'] and pilot['copies']==64 and pilot['provenance_sha256']==sha(P/'provenance.json') and pilot['archive_sha256']==sha(P/'pilot.jsonl.gz') and s['pilot_sha256']==sha(P/'pilot.json')
assert pilot['driver_sha256']==sha(P/'pilot.py') and s['driver_sha256']==sha(P/'summarize_pilot.py')
assert [r['copy'] for r in pilot['rows']]==list(range(64))
old=read(D/'k30_s1_noise20.json');control=read(R/'trained_seed1.json');totals=Counter()
with gzip.open(P/'pilot.jsonl.gz','rt') as f:raw=[json.loads(line) for line in f]
assert len(raw)==64
for j,(r,a) in enumerate(zip(pilot['rows'],raw)):
 saved=old['rows'][j];assert a['copy']==j and r['model_sha256']==a['model_sha256']==saved['model_hash'] and r['truth']==saved['saved_truth'] and r['reused_control']==(j==0)
 b,t=a['baseline'],a['traced']
 for key in ('hex','legacy','stats','info'):assert b[key]==t[key]
 num,ints=ledger(t['ledger']);assert (num,ints)==(r['numeric'],r['integer'])
 assert r['value']==b['value']==saved['recipes']['r85b']['value'] and r['hex']==float(r['value']).hex()==b['hex']
 assert r['legacy']==b['legacy']==saved['recipes']['r85b']['native_ops'] and r['cells']==b['stats']['cells']==saved['recipes']['r85b']['cells']
 totals['numeric']+=num;totals['integer']+=ints
 if j==0:assert b==control['baseline'] and t==control['traced']
assert len({r['model_sha256'] for r in pilot['rows']})==64
ratios={}
for conv,sampler,key in [('named_numeric',111,'numeric'),('historical',72,'legacy')]:
 g=s['groups'][conv];ps=[F(r['truth']) for r in pilot['rows']];es=[(F(r['value'])-p)**2 for r,p in zip(pilot['rows'],ps)];ops=[r[key] for r in pilot['rows']];ns=[o//sampler for o in ops];vs=[p*(1-p) for p in ps];den=[v/n for v,n in zip(vs,ns)];mean=lambda x:sum(x,F())/64;mse=mean(es)
 assert float(mse)==g['mse'] and float(mean(den))==g['mean_sampling_variance_integer'] and float(mse/mean(den))==g['ratio_integer_budget']
 assert float(mse/mean(v*sampler/o for v,o in zip(vs,ops)))==g['ratio_continuous_budget'] and float(mse/(mean(vs)*sampler/mean(map(F,ops))))==g['ratio_mean_budget']
 assert sum(e>d for e,d in zip(es,den))==g['individual_losses']==0
 assert min(ns)==g['min_samples'] and max(ns)==g['max_samples'] and float(mean(map(F,ops)))==g['mean_ops']
 ix=np.random.default_rng(0).integers(0,64,(1000,64));boot=np.array(list(map(float,es)))[ix].mean(1)/np.array(list(map(float,den)))[ix].mean(1)
 assert np.array_equal(np.percentile(boot,[2.5,97.5]),g['bootstrap95_integer']);ratios[conv]=g['ratio_integer_budget']
assert totals['numeric']==s['numeric'] and totals['integer']==s['integer']
assert s['pilot_cpu']==pilot['cpu'] and s['projected_64_shard_cpu']==64*pilot['cpu']
out=dict(complete=True,models=64,totals=dict(totals),ratios=ratios,pilot_cpu=pilot['cpu'],projected_total_cpu=s['projected_64_shard_cpu'],pins={str(P/n):sha(P/n) for n in ['pilot.json','pilot.jsonl.gz','pilot_summary.json','provenance.json','pilot.py','shard.py','FULL_PROTOCOL.md']},review_cpu=time.process_time()-start,scope='All64 saved site ledgers, provenance/parity/reuse and exact own-budget summaries pass. Same original4096-only continuation source preserves source/model/truth and fixed resource gates. Projection is research effort, no production runtime or full-panel performance guarantee; numeric111 and historical72 remain separate.')
Path(__file__).with_name('k30_native_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
