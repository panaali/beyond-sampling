"""Contract complete saved component ledgers, with partial status explicit."""
import hashlib,json,math
from collections import Counter
from fractions import Fraction
from pathlib import Path
HERE=Path(__file__).resolve().parent;inv=json.loads((HERE/'inventory.json').read_text());native=Path(inv['native']);prior=Path(inv['prior'])
for path,sha in inv['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
rows=[json.loads((native/'component.json').read_text())]
for p in sorted((HERE/'components').glob('*.json')):
 r=json.loads(p.read_text())
 if r['complete']:rows.append(r)
indices=[r.get('index',0) for r in rows];assert len(indices)==len(set(indices)) and all(0<=i<48 for i in indices)
summary=[]
for i,r in zip(indices,rows):
 expected=json.loads((prior/'noise_preflight.json' if i==0 else prior/'noise_components'/f'{i:02d}.json').read_text())
 assert r['complete'] and r['model_sha256']==inv['model_sha256'] and r['raw_error_component'].hex()==expected['raw_error_component'].hex() and r['legacy_ops']==expected['legacy_ops'] and r['stats']==expected['stats']
 l=r['ledger'];total=Counter()
 for site,counts in l['sites'].items():total.update(counts)
 assert dict(total)==l['totals'];numeric=sum(v for k,v in total.items() if not k.startswith('integer_control'));integer=sum(v for k,v in total.items() if k.startswith('integer_control'))
 assert numeric==l['named_numeric_total'] and integer==l['integer_control_total'] and numeric+integer==l['inclusive_explicit_total']
 diag=sum(sum(v.values()) for site,v in l['sites'].items() if "stats['traces'] += 1" in site or "stats['pieces'] += 1" in site)
 summary.append({'index':i,'numeric':numeric,'integer':integer,'diagnostic':diag,'visible_after_diagnostic':numeric+integer-diag,'legacy':r['legacy_ops'],'cpu':r['cpu_seconds'],'opaque_calls':l['opaque_calls']})
raw=float(sum((Fraction(r['raw_error_component']) for r in rows),Fraction()))
out={'complete':indices==list(range(48)),'indices':indices,'rows':summary,'numeric':sum(r['numeric'] for r in summary),'integer':sum(r['integer'] for r in summary),'diagnostic':sum(r['diagnostic'] for r in summary),'visible_after_diagnostic':sum(r['visible_after_diagnostic'] for r in summary),'legacy':sum(r['legacy'] for r in summary),'cpu':sum(r['cpu'] for r in summary),'raw_exact_saved_float_sum':raw,'scope':'Partial component census until all48; cold setup and opaque/runtime categories separate. No transferred ratios or full implementation-cost claim.'}
if out['complete']:
 out['estimate']=1-raw/(2*math.pi**2)
 saved=json.loads((prior/'noise_summary.json').read_text());assert out['legacy']==saved['legacy_ops']
 out['original_order_estimate']=1-sum(r['raw_error_component'] for r in rows)/(2*math.pi**2)
 assert out['original_order_estimate']==saved['estimate']
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('rows','indices')}))
