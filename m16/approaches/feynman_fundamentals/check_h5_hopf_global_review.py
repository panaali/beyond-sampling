"""Contract five immutable independent seed audits against full H5 summary."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_native');H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
start=time.process_time();s=read(P/'panel_summary.json');assert s['complete'] and s['source_sha256']==sha(P/'panel_summary.py')
for path,h in read(P/'summary_scope.json')['pins'].items():assert sha(path)==h
rows=[];audits=[]
for seed in range(5):
 path=H/f'h5_hopf_seed{seed}_review_results.json';a=read(path);assert a['complete'] and a['seed']==seed and len(a['rows'])==5;audits.append(sha(path));rows.extend(a['rows'])
assert len(s['rows'])==len(rows)==25 and len({r['model_sha256'] for r in rows})==25
for i,(a,b) in enumerate(zip(rows,s['rows'])):
 assert (a['index'],a['seed'],a['copy'])==(i,i//5,i%5)
 for key in ('index','seed','copy','model_sha256','value','value_hex','legacy','explicit','conditional','closures','closure_count','cpu_seconds','meta_sha256','ledger_sha256','pilot_reused_once'):assert a[key]==b[key],(i,key)
 meta=P/('pilot.json' if i==0 else f'panel/{i:02d}.json');raw=P/('pilot_ledger.json.gz' if i==0 else f'panel/{i:02d}_ledger.json.gz')
 assert sha(meta)==a['meta_sha256'] and sha(raw)==a['ledger_sha256'];d=read(meta)
 assert b['baseline_cpu_seconds']==d['baseline_cpu_seconds'] and b['max_rss_kib']==d['max_rss_kib']
assert sum(r['pilot_reused_once'] for r in rows)==1
for key in ('legacy','explicit','conditional','closure_count','cpu_seconds'):assert s['totals'][key]==sum(r[key] for r in rows)
assert s['new24_cpu_seconds']==sum(r['cpu_seconds'] for r in rows[1:])
out=dict(complete=True,models=25,closure_records=s['totals']['closure_count'],totals=s['totals'],seed_audits=audits,pilot_reused_once=True,summary_sha256=sha(P/'panel_summary.json'),review_cpu=time.process_time()-start,new_evaluator_calls=0,scope='All25 complete original cases and1,433,600 closure records covered by immutable independent raw seed audits. Global source/model/output/hash/category reconciliation passes. Conditional source-cost convention only; old75-family truth uncertainty and original mask defect remain, comparison separately gated.')
assert out['closure_records']==1433600
(H/'h5_hopf_global_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
