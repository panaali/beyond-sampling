import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,time,re
import numpy as np
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native');start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(v):
 if isinstance(v,dict):
  if v.get('array'):
   a=np.array([float.fromhex(x) for x in v['hex']],dtype=v['dtype']).reshape(v['shape']);assert sha_bytes(a.tobytes())==v['bytes_sha256']
  else:
   for x in v.values():arrays(x)
 elif isinstance(v,list):
  for x in v:arrays(x)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
c=read(P/'capture.json');assert c['complete'] and c['all_assigned_objects_bitwise_equal'] and not c['forbidden_model_io_or_clipping_calls'];assert sha(P/'capture.py')==c['runner_sha256'] and sha(P/'pins.json')==c['pins_sha256'];counts=Counter(r['name'] for r in c['rows']);assert counts==dict(inv=48,svd=6,leggauss=6);assert [r['args']['items'][0] for r in c['rows'] if r['name']=='leggauss']==[6,3,3,6,10,16]
arrays(c['objects']);arrays(c['rows']);d=read(P/'inverse_capture.json');assert d['complete'] and d['dispatch_restored'] and d['runtime_targets_and_parameters_match'] and d['capture_sha256']==sha(P/'capture.json');assert d['spy_source_sha256']==sha(P/'inverse_spy.c') and d['spy_binary_sha256']==sha(P/'inverse_spy.so') and d['runtime_expected_sha256']==sha(P/'inverse_runtime_expected.json');assert len(d['rows'])==54 and all(r['parity'] for r in d['rows']);arrays(d['rows'])
orig=[r for r in c['rows'] if r['name']=='inv'];s=read(P/'inverse_summary.json');assert s['complete'] and not s['full_implementation_cost'] and s['capture_sha256']==sha(P/'inverse_capture.json');assert len(s['rows'])==48
for i,(r,old,summary) in enumerate(zip(d['rows'][:48],orig,s['rows'])):
 assert r['name']==f'cold_{i}' and r['input']==old['args']['items'][0] and r['output']==old['output'];ev=Counter((e['kernel'],tuple(e['args'])) for e in r['events']);assert Counter(tuple(e['args'][:3]) for e in r['events'] if e['kernel']=='gemm')==Counter({(1,2,2):1,(1,1,2):1,(2,2,1):1,(2,1,1):1})
 assert [e['args'][0] for e in r['events'] if e['kernel']=='iamax']==[3,2,1];assert [(e['args'][0],e['args'][1]) for e in r['events'] if e['kernel']=='gemv']==[(3,0),(2,1),(1,2)];assert [e['args'] for e in r['events'] if e['kernel']=='dot']==[[1,3,1,0]]
 for kind in ('upper','lower'):assert [e['args'][:3] for e in r['events'] if e['kernel']==kind]==[[3,2,3],[3,1,3]]
 gemm=sum(2*m*n*(k+1) for m,n,k in [(1,2,2),(1,1,2),(2,2,1),(2,1,1)]);local=2*3*(2**2+1**2);assert gemm==42 and local==30
 expected=dict(LU_arithmetic=24,LU_abs_comparison=63,local_solves=local,cross_block_GEMM=gemm,upper_reciprocals=3,solve_guards=4,numeric_upper_subset=166)
 for key,value in expected.items():assert summary[key]==value
assert s['cold_inverse_numeric_upper_subset']==48*166==7968
for path,h in read(P/'inverse_price_pins.json').items():assert sha(Path(path))==h
out=dict(cold_calls=dict(counts),inverse_controls=54,priced_saved_inverses=48,local_solve_per_call=30,tiny_packed_gemm_per_call=42,numeric_subset_per_call=166,total_inverse_subset=7968,review_cpu=time.process_time()-start,scope='Saved argument/output byte encodings, passthrough records/runtime guards and every48cold child-shape contraction checked. New baseline equality is source-asserted; not prior serialized-target parity. Tiny packed FMA widths and2+1solve source counted separately from reviewed3x3LU; full-vectorSVD,leggauss,native/control/memory/setup costs remain open.')
Path(__file__).with_name('cold_inverse3_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
