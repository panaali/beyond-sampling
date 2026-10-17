"""Independent saved T184 binding audit; no model reconstruction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_78/review/recipe112_core_native')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
start=time.process_time();m=read(P/'manifest.json');r=read(P/'provenance.json');g=read(P/'provenance_gate.json')
for name,h in g.items():assert sha(P/name)==h
for path,h in m['pins'].items():assert sha(path)==h
assert r['complete'] and r['cases']==3840 and r['current_model_regenerations']==r['current_estimator_reference_MC_calls']==0
assert r['driver_sha256']==sha(P/'provenance.py') and r['manifest_sha256']==sha(P/'manifest.json') and r['independent_model_hash_regeneration_reused']=='88490d79'
assert r['checkpoints']==m['checkpoints'];cp={(c['H'],c['zoo']):c for c in m['checkpoints']};assert set(cp)=={(h,z) for h in (4,5,6) for z in range(5)}
for c in cp.values():assert sha(c['path'])==c['sha256'] and m['pins'][c['path']]==c['sha256']
source=Path(m['source_root']);native=Path(m['native_root']);binding=read(m['prior_trace_binding'])
assert binding['complete'] and len(binding['rows'])==64 and binding['trace_sha256']==sha(native/'trace.py') and binding['source_sha256']==sha(source/'dependencies/understand_m43/est75.py')
for path,h in read(native/'inventory.json')['pins'].items():assert sha(path)==h
panels={h:read(source/'snapshot/out'/f'fresh_H{h}_abs0.05_80-81-82-83.json') for h in (4,5,6)}
seen=set();models=set();cases=0
for spec,out in zip(m['shards'],r['shards']):
 key=spec['H'],spec['zoo'],spec['noise'];h,z,n=key;assert key not in seen;seen.add(key)
 assert (out['H'],out['zoo'],out['noise'])==key and out['copies']==64 and out['original_rows_bound']
 path=P/spec['snapshot'];assert sha(path)==spec['sha256']==out['snapshot_sha256'];d=read(path)
 assert d['complete'] and d['all_parity_pass'] and len(d['rows'])==d['expected']==64 and not d['borderline']
 assert (d['H'],d['zoo'],d['noise'])==key and d['checkpoint_sha256']==cp[h,z]['sha256']==out['checkpoint_sha256']
 panel=panels[h];assert panel['H']==h and panel['mode']=='abs' and panel['sigma']==.05 and panel['n']==64 and panel['seeds']==[80,81,82,83]
 assert d['source_sha256']==sha(source/'snapshot/out'/f'fresh_H{h}_abs0.05_80-81-82-83.json')
 raw=panel['zoo'][str(z)]['rows'];assert len(raw)==257
 for i,v in enumerate(d['rows']):
  orig=raw[1+64*(n-80)+i];assert v['copy']==i and v['parity_pass'] and v['cell_count_equal']
  assert [v[k] for k in ('truth','saved_estimate','saved_ops','saved_cells')]==[orig[k] for k in ('truth','est','ops','cells')]
  digest=v['model_sha256'];assert len(digest)==64 and int(digest,16)>=0 and digest not in models;models.add(digest);cases+=1
assert cases==len(models)==3840 and seen=={(h,z,n) for h in (4,5,6) for z in range(5) for n in range(80,84)}
out=dict(complete=True,cases=cases,shards=len(seen),checkpoints=len(cp),prior_independent_regeneration='88490d79',provenance_sha256=sha(P/'provenance.json'),manifest_sha256=sha(P/'manifest.json'),review_cpu=time.process_time()-start,new_model_regeneration_or_evaluator_calls=0,scope='Independent byte/source and original saved-row binding only; prior independently regenerated model hashes reused. No multiplier transfer or new truth/cost/accuracy evidence.')
Path(__file__).with_name('core_native_provenance_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
