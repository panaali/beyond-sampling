"""Saved pilot/source provenance; never draw random samples."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,json,math,ast,zlib
import numpy as np
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_reference_precision')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(P/'protocol.json');c=read(P/'controls.json');b=read(P/'s0_c0/block000.json');r=read(P/'pilot_review.json')
for row in p['files'].values():assert sha(Path(row['path']))==row['sha256']
assert len(c['records'])==6 and c['extra_forward_evaluations']==318
assert {(x['n'],x['chunk']) for x in c['records']}=={(n,k) for n in (5,17,31) for k in (3,11)}
for x in c['records']:assert 0<=x['hits']<=x['n'] and x['value']==x['hits']/x['n']
models=read(Path(p['files']['models']['path']))['rows']
assert len(p['models'])==len(models)==25
seeds=[]
for ident in p['models']:
    row=next(x for x in models if (x['seed'],x['copy'])==(ident['seed'],ident['copy']))
    arrays=[np.asarray(w,dtype='<f8') for w in row['model']]
    h=hashlib.sha256(b''.join(w.tobytes() for w in arrays)).hexdigest()
    assert h==ident['model_sha256']==row['model_sha256'];seeds.append(ident['rng_seed'])
assert len(set(seeds))==25
ident=p['models'][0];assert ident['seed']==ident['copy']==0
assert b['status']=='complete' and b['block']==b['seed']==b['copy']==0 and b['n']==1_000_000
assert b['model_sha256']==c['model_sha256']==ident['model_sha256']
assert b['protocol_sha256']==sha(P/'protocol.json') and b['source_sha256']==sha(P/'reference.py')
assert b['state_before']==np.random.default_rng(ident['rng_seed']).bit_generator.state
assert b['state_after']['bit_generator']=='PCG64' and b['state_after']['state']['inc']==b['state_before']['state']['inc']
assert isinstance(b['hits'],int) and 0<=b['hits']<=b['n']
assert r['pilot_sha256']==sha(P/'s0_c0/block000.json')
assert r['fixed_total_draws']==2_500_000_000 and r['pilot_draws']==b['n']
projected=b['cpu_seconds']*2500
assert math.isclose(projected,r['projected_reference_cpu_seconds'],rel_tol=1e-15)
assert projected<=r['frozen_projection_limit']==600 and r['projection_gate_pass'] and not r['accuracy_used_for_gate']
out=dict(complete=True,model_hashes=25,deterministic_cases=6,extra_evaluations=318,
         pilot_draws=b['n'],projected_cpu_seconds=projected,pilot_sha256=sha(P/'s0_c0/block000.json'),
         continuation_sha256=sha(P/'continue_reference.py'),random_draws_in_review=0,
         scope='The fixed throughput gate passes; projection is not a runtime guarantee. Frozen 100M per model, original block0 reused once; no accuracy-based stopping. Final hits/RNG continuity and confidence calculation remain pending.')
(HERE/'hopf_reference_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
