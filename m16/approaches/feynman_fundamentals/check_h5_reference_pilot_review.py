"""Saved fixed pilot metadata/state/throughput audit; no random draws."""
from pathlib import Path
import hashlib,json,resource,time
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_reference');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((P/'pilot.json').read_text());p=json.loads((P/'protocol.json').read_text());c=json.loads((P/'controls.json').read_text());r=json.loads((O/'h5_reference_controls_review_results.json').read_text())
for scope in ('source_scope.json','pilot_scope.json'):
 for f,h in json.loads((P/scope).read_text())['pins'].items():assert sha(Path(f))==h
assert r['pass'] and r['control_sha256']==sha(P/'controls.json') and c['complete']
assert d['complete'] and d['phase']=='complete' and (d['index'],d['block'],d['seed'],d['copy'])==(0,0,0,0)
ident=p['models'][0]
for key in ('model_sha256','rng_seed','rng_name'):assert d[key]==ident[key]
assert d['source_sha256']==sha(P/'reference.py') and d['protocol_sha256']==sha(P/'protocol.json')
assert d['n']==p['block_size']==1000000 and type(d['hits'])is int and 0<=d['hits']<=d['n']
assert d['state_before']==np.random.PCG64(d['rng_seed']).state
s=d['state_after'];assert s['bit_generator']=='PCG64' and s['state']['inc']==d['state_before']['state']['inc'] and s['state']['state']!=d['state_before']['state']['state'] and 0<=s['state']['state']<2**128 and s['has_uint32']==0
assert d['projected_full_forward_cpu_seconds']==d['forward_cpu_seconds']*p['blocks_per_model']*len(p['models']) and d['projection_gate_pass'] and d['projected_full_forward_cpu_seconds']<=p['total_projection_gate_forward_cpu_seconds']==900
assert 0<d['forward_cpu_seconds']<=d['cpu_seconds'] and d['max_rss_kib']<2000000
out={'pass':True,'pilot_sha256':sha(P/'pilot.json'),'control_sha256':sha(P/'controls.json'),'fixed_draws':d['n'],'hits':d['hits'],'forward_cpu':d['forward_cpu_seconds'],'projected_full_cpu':d['projected_full_forward_cpu_seconds'],'review_cpu':time.process_time()-start,'review_random_draws':0,'scope':'Initial PCG state reconstructed without drawing; final state schema/increment and saved pins verified, not replayed. Throughput scheduling gate passes; no accuracy-based selection. Continuation must reuse pilot once and preserve every block state/hit. Full expansion still needs source gate.'}
(O/'h5_reference_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
