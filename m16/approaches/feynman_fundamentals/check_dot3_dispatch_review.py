import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import json,hashlib
from collections import Counter
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native/outer_coverage/dot3')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(P/'results.json');p=read(P/'protocol.json');b=read(P/'build.json')
assert r['complete'] and r['restored'] and r['control_count']==90 and r['all_bitwise_parity']
assert r['protocol_sha256']==sha(P/'protocol.json') and r['source_sha256']==sha(P/'run.py') and r['spy_build']==b
assert sha(P/'spy.c')==b['source_sha256'] and sha(P/'spy.so')==b['binary_sha256']
for path,h in p['pins'].items():assert sha(path)==h
runtime=r['runtime'];assert runtime['dot_offset']=='0xa3d7c0' and runtime['slot_offset']=='0x368' and runtime['rounding']==0
assert int(runtime['dot_target'],16)-int(runtime['blas_base'],16)==0xa3d7c0
seen=set();events=Counter();by=Counter()
for row in r['rows']:
 key=(row['pattern'],row['layout'],row['operation']);assert key not in seen;seen.add(key)
 assert row['parity'] and row['value_bits']==row['baseline_bits']
 vals=p['patterns_hex'][row['pattern']];other=[vals[2],vals[0],vals[1]]
 if row['layout']=='reversed':vals=vals[::-1];other=other[::-1]
 assert row['input_a_hex']==vals and row['input_b_hex']==other
 stride={'contiguous':8,'stride2':16,'reversed':-8}[row['layout']]
 assert row['strides_a']==row['strides_b']==[stride]
 if row['operation']=='matmul' and stride<0:expected=[]
 else:
  inc=2 if stride==16 and row['operation']!='norm' else 1
  expected=[dict(n=3,incx=inc,incy=inc)]
 assert row['events']==expected
 events[str(expected)]+=1;by[str((row['operation'],row['layout']))]+=1
assert seen=={(i,l,o) for i in range(10) for l in p['layouts'] for o in p['operations']}
manifest=read(P/'parent_disassembly_manifest.json')
assert sha(manifest['binary'])==manifest['binary_sha256'] and sha(manifest['artifact'])==manifest['artifact_sha256']
out=dict(complete=True,cases=90,event_shapes=dict(events),operation_layout_counts=dict(by),results_sha256=sha(P/'results.json'),candidate_numeric_subset=dict(contiguous_dot=7,contiguous_norm=8,positive_nonunit_dot=8,scalar_fallback_dot=6),scope='Saved parity/runtime/restore and all90 layout events checked. Installed n3 arithmetic tails support selected subset counts under FMA=2; dtype/shape applicability and historical binding remain separate. No historical corpus price replacement or native/hardware total inferred.',new_kernel_calls=0)
Path(__file__).with_name('dot3_dispatch_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
