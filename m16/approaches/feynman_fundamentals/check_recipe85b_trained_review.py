import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_910/review/recipe85b_native_cost');T=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery');start=time.process_time()
def read(p):return json.loads(p.read_text())
manifest=read(T/'continuation_manifest.json');out=[]
for seed,noise,copy in [(1,20,0),(4,23,255)]:
 d=read(P/f'trained_seed{seed}.json');assert (d['seed'],d['noise'],d['copy'])==(seed,noise,copy)
 for path,digest in d['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 old=read(T/f'k30_s{seed}_noise{noise}.json')['rows'][copy];assert d['saved']==old and d['model_sha256']==old['model_hash'];checkpoint=manifest[f'checkpoint_{seed}'];assert d['checkpoint_sha256']==checkpoint['sha256']==hashlib.sha256(Path(checkpoint['source']).read_bytes()).hexdigest()
 a,b=d['baseline'],d['traced'];assert all(a[k]==b[k] for k in ['hex','legacy','stats','info']);expected=old['recipes']['r85b'];assert a['value']==expected['value'] and a['legacy']==expected['native_ops'] and a['stats']['cells']==expected['cells'];c=Counter()
 for v in b['ledger']['sites'].values():c.update(v)
 assert dict(c)==b['ledger']['totals'];ints=sum(v for k,v in c.items() if k.startswith('integer_control'));numeric=sum(c.values())-ints;assert numeric==b['ledger']['named_numeric_total'] and ints==b['ledger']['integer_control_total'];assert d['sampler_numeric']==111
 out.append(dict(seed=seed,value=a['value'],legacy=a['legacy'],named_numeric=numeric,separate_integer=ints,sort_calls=b['ledger']['sort_dispatch_calls'],author_cpu=d['cpu']))
assert [r['named_numeric'] for r in out]==[114665,92227] and [r['sort_calls'] for r in out]==[40,32]
result=dict(cases=out,review_cpu=time.process_time()-start,scope='Two frozen T115 cases only; source/checkpoint/model-hash links to independently reviewed complete panel, every site sum and parity checked. No native rerun, multiplier transfer or full training-rung/common hardware-cost claim.')
Path(__file__).with_name('recipe85b_trained_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
