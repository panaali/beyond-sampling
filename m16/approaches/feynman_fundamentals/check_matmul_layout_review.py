import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import json,hashlib,zipfile,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/log_cofactor_cost/matmul');start=time.process_time();d=json.loads((P/'layout_applicability.json').read_text());assert d['complete'];rays=Path(d['rays_path']);assert hashlib.sha256(rays.read_bytes()).hexdigest()==d['rays_sha256']
with zipfile.ZipFile(rays) as z:
 with z.open('rays.npy') as f:
  assert np.lib.format.read_magic(f)==(1,0);shape,order,dtype=np.lib.format.read_array_header_1_0(f)
assert shape==(35098,4,4) and not order and dtype==np.dtype('float64');assert d['indexed_rays_layout']['strides']==[32,8]
assert hashlib.sha256((P.parent/'frozen/estT_snapshot.py').read_bytes()).hexdigest()==d['source_sha256'];assert d['du_layout']['shape']==[1000,4] and d['du_layout']['strides']==[32,8] and d['du_layout']['C'];assert d['cofactor_N_allocation']['strides']==[32,8] and d['cofactor_N_allocation']['C']
assert len(d['synthetic_inverses'])==2
for i,r in enumerate(d['synthetic_inverses']):
 assert r['input_order']==('C','F')[i] and r['input']['strides']==([32,8],[8,32])[i] and r['transposed_input']['strides']==([8,32],[32,8])[i] and r['output']['strides']==[32,8] and r['output']['C'] and r['exact_identity']
assert (d['new_synthetic_inverse_calls'],d['model_free_duffy_setups'],d['new_matrix_products'],d['new_mass_model_MC_calls'])==(2,1,0,0)
subtotal=72000*17955+170*(35098+17143);assert subtotal==1301640970
revision=json.loads((P/'revised_summary.json').read_text());old=json.loads((P.parent/'summary.json').read_text());assert hashlib.sha256((P.parent/'summary.json').read_bytes()).hexdigest()==revision['old_summary_sha256']
removed=28000*17955+112*(35098+17143);guards=11*(35098+17143);assert (removed,subtotal,guards)==(revision['removed_scalar_matrix_terms'],revision['executed_matrix_body_subset'],revision['selected_matrix_guard_envelopes']);assert revision['revised_hybrid_subtotal']==revision['old_hybrid']-removed+subtotal+guards==3572147383;assert revision['model_free_runtime_constants_unchanged']==190197;assert not revision['complete_native_upper_or_lower'] and not revision['same_budget_comparison']
for key,n,base,remove,body,g in [('thin',17955,38617,28112,72170,11),('wide',17143,121633,224,340,22)]:
 r=revision['branches'][key];assert (r['copies'],r['old_hybrid'],r['removed_scalar_matrix_terms'],r['replacement_matrix_body'],r['replacement_guard_envelope'])==(n,base,remove,body,g);assert r['revised_hybrid']==base-remove+body+g
out=dict(revised_hybrid=revision['revised_hybrid_subtotal'],fixed_replay_layout_passed=True,thin=17955,all_cones=35098,wide=17143,matrix_body_lane_subset=subtotal,inherited_nominal_mass_counter=1064714168,review_cpu=time.process_time()-start,scope='Raw archive header/hash and saved allocation evidence checked; no native/mass/model replay. Applies only fixed captured C-layout replay and pinned NumPy/backend. Executed FMA=2 lane-body subset exceeds inherited nominal mass units, but conventions differ; no full runtime or sampler comparison.')
Path(__file__).with_name('matmul_layout_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
