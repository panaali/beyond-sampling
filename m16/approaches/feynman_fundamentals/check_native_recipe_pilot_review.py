import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,time
P=Path('/workspace/work/recover-small_h2_910/review/recipe112_native_cost');start=time.process_time();d=json.loads((P/'panel/H8_z0_n80.json').read_text());archive=P/'panel'/d['archive'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==d['archive_sha256'];full=[json.loads(line) for line in gzip.decompress(archive.read_bytes()).splitlines()];priorpath=P.parent/'recipe112_recovery/direct_counts/core_H8_z0_n80.json';assert hashlib.sha256(priorpath.read_bytes()).hexdigest()==d['prior_sha256'];prior=json.loads(priorpath.read_text());assert d['complete'] and len(full)==len(d['rows'])==len(prior['rows'])==64
for path,key in [(P/'trace.py','trace_sha256'),(P/'panel.py','driver_sha256'),(P.parent/'recipe112_recovery/dependencies/understand_m43/est75.py','source_sha256')]:assert hashlib.sha256(path.read_bytes()).hexdigest()==d[key]
total=integer=0
for i,(r,a,p) in enumerate(zip(d['rows'],full,prior['rows'])):
 assert r['copy']==a['copy']==p['copy']==i and r['model_sha256']==a['model_sha256']==p['model_sha256'];b,t=a['baseline'],a['traced'];assert all(b[k]==t[k] for k in ['hex','legacy','stats','info']);assert b['value']==r['estimate'] and abs(r['estimate']-p['estimate'])<=1e-12;assert r['saved_scalar_ops']==p['ops'];assert r['legacy']==b['legacy']==p['saved_ops'];assert r['cells']==b['stats']['cells']==p['stats']['cells'];c=Counter()
 for counts in t['ledger']['sites'].values():c.update(counts)
 assert dict(c)==t['ledger']['totals']==r['totals'];ints=sum(v for k,v in c.items() if k.startswith('integer_control'));assert ints==r['integer']==t['ledger']['integer_control_total'];num=sum(c.values())-ints;assert num==r['ops']==t['ledger']['named_numeric_total'];total+=num;integer+=ints
assert total==30628550 and d['sampler_numeric']==351
out=dict(cases=64,site_ledgers_checked=64,model_hashes_match_independently_reviewed_direct_panel=True,named_numeric_total=total,integer_categories=integer,author_cpu=d['cpu'],review_cpu=time.process_time()-start,scope='One frozen64-case pilot; no native trace rerun or multiplier transfer. Named numeric sampler351 excludes integer correctness and implicit/compiled/library costs remain qualified.')
Path(__file__).with_name('native_recipe_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
