"""Original-order contraction of one completed frozen noisy case."""
import hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
identity=json.loads((HERE/'noise_identity.json').read_text())
rows=[json.loads((HERE/'noise_preflight.json').read_text())]
rows += [json.loads((HERE/'noise_components'/f'{i:02d}.json').read_text()) for i in range(1,48)]
observer_sha=hashlib.sha256((HERE/'observer.py').read_bytes()).hexdigest();runner_sha=hashlib.sha256((HERE/'noise_components.py').read_bytes()).hexdigest()
total=0.0
for i,r in enumerate(rows):
 assert r['complete'] and r['model_sha256']==identity['model_sha256'] and r['observer_sha256']==observer_sha
 assert (r['family'],r['side'])==divmod(i,2)
 if i:assert r['index']==i and r['runner_sha256']==runner_sha
 assert r['stats']['traces']==r['observer'].get('trace',{}).get('calls',0)
 assert math.isfinite(r['raw_error_component'])
 total+=r['raw_error_component']
counts={}
for r in rows:
 for name,d in r['observer'].items():
  target=counts.setdefault(name,{})
  for k,v in d.items():target[k]=max(target.get(k,0),v) if k.startswith('max_') else target.get(k,0)+v
out={'model_sha256':identity['model_sha256'],'complete_components':len(rows),'estimate':1-total/(2*math.pi**2),'raw_error_sum':total,'legacy_ops':sum(r['legacy_ops'] for r in rows),'stats':{k:sum(r['stats'][k] for r in rows) for k in ('traces','pieces')},'observer':counts,'guard':{k:sum(r['guard'][k] for r in rows) for k in ('calls','exact_zero')},'component_cpu_seconds':sum(r['cpu_seconds'] for r in rows),'max_component_cpu_seconds':max(r['cpu_seconds'] for r in rows),'max_rss_kib':max(r['max_rss_kib'] for r in rows),'source_scope':'Original F2L3 plus reviewed exact-zero guard and cap observer, unchanged cubature. Legacy ops are not complete cost. No independent truth yet, no rung claim.'}
(HERE/'noise_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
