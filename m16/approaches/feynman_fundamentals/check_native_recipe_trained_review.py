import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time,numpy as np,torch
P=Path('/workspace/work/recover-small_h2_910/review/recipe112_native_cost');R=P.parent/'recipe112_recovery';start=time.process_time();semantic=json.loads((R/'semantics.json').read_text());results=[]
for H in [4,8]:
 d=json.loads((P/f'trained_H{H}.json').read_text());old=next(r for r in semantic['models'] if r['H']==H);path=Path(old['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==d['checkpoint_sha256'];st=torch.load(path,map_location='cpu',weights_only=True);base=tuple(a.numpy().astype(float) for a in [st['rnn.weight_ih_l0'][:,0],st['rnn.weight_hh_l0'],st['linear.weight']]);rng=np.random.default_rng(89)
 for _ in range(d['copy']+1):m=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
 assert hashlib.sha256(b''.join(w.tobytes() for w in m)).hexdigest()==d['model_sha256']
 for name,sha in d['source_hashes'].items():
  path=R/'dependencies/understand_m43/est75.py' if name=='est75.py' else P/name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 a,b=d['baseline'],d['traced'];assert all(a[k]==b[k] for k in ['hex','legacy','stats','info']);assert len(b['info'])==12 and len({tuple(x['key']) for x in b['info']})==12
 counts=Counter()
 for c in b['ledger']['sites'].values():counts.update(c)
 assert dict(counts)==b['ledger']['totals'];integer=sum(v for k,v in counts.items() if k.startswith('integer_control'));numeric=sum(counts.values())-integer;assert integer==b['ledger']['integer_control_total'] and numeric==b['ledger']['named_numeric_total']
 scalar=next(r for r in json.loads((R/f'counter_H{H}.json').read_text())['rows'] if (r['zoo'],r['copy'])==(d['zoo'],d['copy']));assert scalar['repaired_counter']==d['scalar_repaired_counter'];assert a['legacy']==scalar['author_counter'];assert abs(a['value']-scalar['author'])<1e-15
 results.append(dict(H=H,named_numeric=numeric,integer_categories=integer,repaired_scalar=scalar['repaired_counter'],legacy=a['legacy'],err_by_entries=12))
out=dict(cases=results,model_hashes_independently_reconstructed=True,review_cpu=time.process_time()-start,scope='Two frozen native traces only; independent saved site sums/parity/provenance. No transferred multiplier, estimator rerun, or full compiled/library/control cost claim.')
Path(__file__).with_name('native_recipe_trained_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
