import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,json,math,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/positive_model_adapter')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time();protocol=read(P/'capture_protocol.json');out=[]
for path,digest in protocol['pins'].items():assert sha(Path(path))==digest
for c in protocol['cases']:
 seed=c['seed'];r=read(P/f'capture_seed{seed}.json');a=np.load(P/f'captured_seed{seed}.npz')['rays'];b=np.load(c['reference_capture'])['rays'];assert np.array_equal(a,b);assert a.dtype==b.dtype==np.float64 and a.shape[1:]==(4,4)
 assert r['complete'] and r['source_sha256']==sha(P/'capture_parity.py') and r['protocol_sha256']==sha(P/'capture_protocol.json')
 assert hashlib.sha256(a.tobytes()).hexdigest()==r['ordered_ray_sha256'];cap=r['capture'];assert cap['N']==len(a) and cap['copied_float64_elements']==a.size and cap['ray_storage_bytes']==a.nbytes
 assert cap['stats']==read(Path(c['capture_metadata']))['stats'];assert read(P/f'captured_seed{seed}.json')['capture']==cap
 g=cap['guard_counts'];assert not any(g.get(k,0) for k in ['earlier_tie','later_tie','integer_orientation_comparisons']);assert sum(g.get(k,0) for k in ['scalar_equalities','boolean_reductions','integer_orientation_comparisons'])==r['additional_guard_numeric_units']
 prior=np.load(c['positive_values']);s=read(Path(c['positive_summary']));assert len(prior['values'])==len(a);assert 1-math.fsum(prior['values'])==s['accuracy']==r['saved_value_diagnostic']['accuracy'];assert math.fsum(prior['bounds'])==r['saved_value_diagnostic']['bound'];assert s['known_float_named_total']==r['saved_value_diagnostic']['known_mass_work_to_charge']
 out.append(dict(seed=seed,ordered_rays=len(a),ray_storage_bytes=a.nbytes,guard_numeric_units=r['additional_guard_numeric_units'],capture_cpu=cap['cpu']))
cost=read(P/'cost_composition.json')
for path,digest in cost['pins'].items():assert sha(Path(path))==digest
assert sum(cost['components'].values())==cost['visible_composed_subset']==185465029;assert cost['additional_48_inverse_numeric_envelope']==48*290==13920;assert cost['subset_with_inverse_numeric_envelope']==185478949;assert cost['components']['unused_original_rule_import_visible']==sum(9*n+1 for n in [12,48]);assert cost['components']['new_guard_numeric']==out[0]['guard_numeric_units']
result=dict(cases=out,seed0_known_composed_subset=185465029,additional_inverse_numeric=13920,review_cpu=time.process_time()-start,scope='Every saved ray compared bitwise, no capture/mass rerun. Prior mass work remains charged; no seed0 geometry-cost transfer toseed1. Failed110CPU validation and generic alloff finite-order unresolved case remain. Known subset excludes named remaining costs and full floating certification.')
Path(__file__).with_name('adapter_capture_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
