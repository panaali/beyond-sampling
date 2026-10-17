"""Three T184 pilots: full saved sites and historical parity, no replay."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_78/review/recipe112_core_native')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ledger(l):
 c=Counter()
 for counts in l['sites'].values():
  assert all(type(v)is int and v>=0 for v in counts.values());c.update(counts)
 assert dict(c)==l['totals'] and not l['complete_implementation_cost']
 integer=sum(v for k,v in c.items() if k.startswith('integer_control'));numeric=sum(c.values())-integer
 assert (numeric,integer)==(l['named_numeric_total'],l['integer_control_total'])
 return numeric,integer,dict(c)
start=time.process_time();d=read(P/'pilots.json');g=read(P/'pilot_gate.json');m=read(P/'manifest.json');prov=read(P/'provenance.json')
assert d['complete'] and d['gate']==g and prov['complete'] and prov['cases']==3840
for path,h in g['pins'].items():assert sha(path)==h
for path,h in m['pins'].items():assert sha(path)==h
assert d['trace_sha256']==sha(Path(m['native_root'])/'trace.py') and d['source_sha256']==sha(Path(m['source_root'])/'dependencies/understand_m43/est75.py')
assert [(r['H'],r['zoo'],r['noise'],r['copy']) for r in d['rows']]==[(h,0,80,0) for h in (4,5,6)]
rows=[]
for r in d['rows']:
 h=r['H'];old=read(P/f'frozen/core/core_H{h}_z0_n80.json');saved=old['rows'][0]
 assert r['stage']=='complete' and r['parity'] and r['saved']==saved
 assert r['model_sha256']==saved['model_sha256'] and r['checkpoint_sha256']==old['checkpoint_sha256']
 b,t=r['baseline'],r['traced']
 for name in ('value','hex','legacy','stats','info'):assert b[name]==t[name]
 assert b['hex']==float(b['value']).hex()==float(saved['saved_estimate']).hex()
 assert b['legacy']==saved['saved_ops'] and b['stats']['cells']==saved['saved_cells']
 numeric,integer,counts=ledger(t['ledger']);sn,si,sc=ledger(t['startup'])
 assert (numeric,integer)==(r['named_numeric'],r['integer_control']) and r['sampler_numeric']==4*h*h+12*h-1
 rows.append(dict(H=h,model_sha256=r['model_sha256'],value_hex=b['hex'],legacy=b['legacy'],numeric=numeric,integer=integer,startup_numeric=sn,startup_integer=si,categories=counts,startup_categories=sc,sampler_numeric=r['sampler_numeric']))
assert d['model_reconstructions']==3 and d['source_estimator_calls']==6 and d['new_truth_or_MC_calls']==0
progress=[json.loads(x) for x in (P/'pilots_progress.jsonl').read_text().splitlines()]
assert len(progress)==6 and [x['stage'] for x in progress]==['attempt','complete']*3
for i,r in enumerate(d['rows']):assert progress[2*i+1]==r
out=dict(complete=True,rows=rows,pilots_sha256=sha(P/'pilots.json'),gate_sha256=sha(P/'pilot_gate.json'),review_cpu=time.process_time()-start,research_cpu=d['cpu_seconds'],new_evaluator_or_model_calls=0,scope='All three fixed saved pilot site/category ledgers, historical bits/counters/cells and baseline/traced grouped contributions pass. Direct declared source convention only, no full native/runtime cost or population accuracy claim. Startup retained separately; full panel requires fresh scope/resource gate.')
Path(__file__).with_name('core_native_pilots_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}));print(json.dumps([{k:r[k] for k in ('H','numeric','integer','legacy','startup_numeric')} for r in rows]))
