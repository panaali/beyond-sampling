"""Complete missing-k30 coverage and independent per-seed summary reconciliation."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,time
HERE=Path(__file__).resolve().parent;P=Path('/workspace/work/recover-small_h4_4/review/margin_k30_native');start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(P/'full_summary.json');assert s['complete'] and s['rows']==4096 and len(s['files'])==64 and len(s['groups'])==4 and len(s['noise_groups'])==16 and len(s['individual'])==4096
assert s['source_sha256']==sha(P/'full_summary.py') and s['provenance_sha256']==sha(P/'provenance.json') and s['pilot_gate_sha256']==sha(P/'pilot_review.json')
assert s['helper_sha256']==read(P/'pilot_summary.json')['helper_sha256']
allindividual=[];files={};totals=Counter();amounts=Counter();reuses=[];reviewpins={}
for seed in range(1,5):
 path=HERE/f'k30_native_seed{seed}_review_results.json';d=read(path);assert d['complete'] and d['rows']==1024 and d['seed']==seed;reviewpins[path.name]=sha(path)
 g=next(x for x in s['groups'] if x['seed']==seed)
 for name,metrics in d['metrics'].items():
  for k,v in metrics.items():assert g['metrics'][name][k]==v,(seed,name,k)
 for noise,metrics in d['noise_metrics'].items():
  g=next(x for x in s['noise_groups'] if x['seed']==seed and x['noise']==int(noise))
  for name,mm in metrics.items():
   for k,v in mm.items():assert g['metrics'][name][k]==v,(seed,noise,name,k)
 allindividual.extend(d['individual']);files.update(d['files']);totals.update(d['totals']);reuses.extend(d['reuses'])
 for k in ('numeric','integer','legacy'):amounts[k]+=d[k]
assert allindividual==s['individual'] and len({r['model_sha256'] for r in allindividual})==4096
assert {(r['seed'],r['noise'],r['copy']) for r in allindividual}=={(a,b,c) for a in range(1,5) for b in range(20,24) for c in range(256)}
assert files==s['files'] and dict(totals)==s['totals'] and reuses==s['reused_controls']==[[1,20,0],[4,23,255]]
assert all(amounts[k]==s[k] for k in amounts)
cpu=0.;maxcpu=0.;rss=0
for relative,h in files.items():
 path=P/relative;assert sha(path)==h;d=read(path);assert d['complete'];cpu+=d['cpu'];maxcpu=max(maxcpu,d['cpu']);rss=max(rss,d['rss_kib'])
assert cpu==s['research_cpu'] and maxcpu==s['max_shard_cpu'] and rss==s['max_rss_kib']
loss={name:sum(r['ratios'][name]>1 for r in allindividual) for name in ('named_numeric','historical')}
out=dict(complete=True,models=4096,full_site_ledgers=4096,shards=64,totals=dict(amounts),individual_losses=loss,research_cpu=cpu,max_shard_cpu=maxcpu,seed_metrics={str(g['seed']):g['metrics'] for g in s['groups']},review_pins=reviewpins,summary_sha256=sha(P/'full_summary.json'),review_cpu=time.process_time()-start,new_model_truth_calls=0,scope='All four independent full-ledger seed audits reconcile exact summaries, sixteen noise groups, all4096 individual budgets, disjoint model hashes and both historical reuses once. Conditional saved-truth/bootstrap and selected source111/historical72 accounting only, no full native/runtime/population guarantee.')
(HERE/'k30_native_global_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('seed_metrics','review_pins')}))
