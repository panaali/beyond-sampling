import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,time
P=Path('/workspace/work/recover-small_h2_910/review/fibre_native_completion');start=time.process_time()
def read(p):return json.loads(p.read_text())
def ledger(l):
 c=Counter()
 for v in l['sites'].values():c.update(v)
 assert c==Counter(l['totals']);return c
outrows=[]
for idx in (2,6):
 folder=P/'cell_rescue'/f'{idx:02d}';prep=read(folder/'prepare.json');s=read(folder/'summary.json');old=read(P/'components'/f'{idx:02d}.json');rows=[read(folder/f'cell_{k:05d}.json') for k in range(prep['cell_count'])];sites={};totals=Counter();prev=0.0;prevhash=None
 for k,r in enumerate(rows):
  assert r['complete'] and r['cell']==k and r['poly']==prep['cells'][k]['poly'] and r['source_sha256']==prep['source_sha256'] and r['model_sha256']==s['model_sha256'];assert r['initial_total'].hex()==prev.hex() and r['previous_sha256']==prevhash;assert r['total_hex']==r['total'].hex() and r['total_type'] in ('float','numpy.float64');assert r['prepare_sha256']==hashlib.sha256((folder/'prepare.json').read_bytes()).hexdigest();prev=r['total'];prevhash=hashlib.sha256((folder/f'cell_{k:05d}.json').read_bytes()).hexdigest()
 assert prev.hex()==old['raw_error_component'].hex()==s['raw_error_component'].hex()
 for r in [prep]+rows:
  totals.update(ledger(r['ledger']))
  for site,c in r['ledger']['sites'].items():sites.setdefault(site,Counter()).update(c)
 assert totals==Counter(s['ledger']['totals']);assert {key:sum(r['stats'][key] for r in rows) for key in ('traces','pieces')}==old['stats']==s['stats'];repeat=(len(rows)-1)*16*prep['clamps'];assert repeat==s['repeated_legacy_setup'];assert prep['legacy_ops']+sum(r['legacy_ops'] for r in rows)==s['actual_adapter_legacy_ops']==old['legacy_ops']+repeat
 diff={}
 for site in set(sites)|set(old['ledger']['sites']):
  a=sites.get(site,Counter());b=Counter(old['ledger']['sites'].get(site,{}));v={k:a[k]-b[k] for k in set(a)|set(b) if a[k]!=b[k]}
  if v:diff[site]=v
 expected={site:{k:v*(len(rows)-1) for k,v in counts.items()} for site,counts in rows[0]['ledger']['sites'].items() if site.startswith('_fn:')}
 assert diff==expected==s['control_site_difference'];assert all(v>=0 for row in diff.values() for v in row.values());outrows.append(dict(index=idx,cells=len(rows),final_hex=prev.hex(),stats=s['stats'],repeated_legacy=repeat,only_repeated_setup_site_differences=True))
assert outrows[0]['stats']['pieces']>0
out=dict(controls=outrows,review_cpu=time.process_time()-start,scope='All61 saved cell-state chains, exact hex continuity, parent-cell order, legacy/stats and complete per-site deltas checked. Nonzero control exercises accumulation; caches are source-local per cell. No component evaluations repeated. Original timeout remains; each new process retains its own cap and transport/import costs are separate.')
Path(__file__).with_name('fibre_cell_rescue_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
