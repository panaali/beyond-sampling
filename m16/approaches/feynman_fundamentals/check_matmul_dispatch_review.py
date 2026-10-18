import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,struct,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/log_cofactor_cost/matmul');start=time.process_time()
def read(p):return json.loads(p.read_text())
b=read(P/'baseline.json');d=read(P/'instrumented.json');assert b['complete'] and d['complete'] and len(b['rows'])==len(d['rows'])==12
D=np.array([[4,1,2,0],[1,5,0,1],[2,0,6,1],[0,1,1,3]],float);I=np.eye(4);FF=np.asfortranarray(D);S=(np.arange(64).reshape(8,8)%7-3).astype(float)[::2,::2];U=(np.arange(4000).reshape(1000,4)%7-3).astype(float);US=(np.arange(8000).reshape(2000,4)%7-3).astype(float)[::2]
fixtures=[(I,I.T),(D,D.T),(FF,FF.T),(S,S.T),(D,D.copy()),(FF,D.copy()),(np.zeros((1000,4)),D),(np.ones((1000,4)),D),(U,D),(U,D.T),(np.asfortranarray(U),D),(US,D)];counts=Counter();manifest=read(P/'manifest.json')
for name,path in manifest['binary_paths'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest['binary_sha256'][name]
for r,o,(A,B) in zip(d['rows'],b['rows'],fixtures):
 assert o=={k:v for k,v in r.items() if k!='events'}
 for key,a in [('A',A),('B',B)]:assert r[key]['sha256_C_values']==hashlib.sha256(a.tobytes(order='C')).hexdigest() and r[key]['shape']==list(a.shape) and r[key]['strides']==list(a.strides)
 expected=[float(sum(int(A[i,k])*int(B[k,j]) for k in range(4))).hex() for i in range(len(A)) for j in range(4)];assert expected==r['output_hex'];assert len(r['events'])==1
 e=r['events'][0];counts['syrk' if e['kind']==1 else 'gemm']+=1;assert e['k']==4 and e['n']==4 and e['m']==len(A);assert struct.unpack('d',struct.pack('q',e['alpha_bits']))[0]==1 and struct.unpack('d',struct.pack('q',e['beta_bits']))[0]==0
 assert e['kind']==(1 if r['name'] in ('identity_self','dense_C_self','dense_F_self') else 0)
 if r['name']=='strided_self':assert r['input_same_pointer'] and not e['same_pointer']
assert counts==Counter(syrk=3,gemm=9)
child=read(P/'kernel_controls.json');assert child['complete'] and child['new_matrix_calls']==12 and child['new_mass_model_MC_calls']==0 and child['binary_sha256']==manifest['binary_sha256'];baseline={r['name']:r for r in b['rows']};childcounts=Counter()
for r in child['rows']:
 assert r['bitwise_baseline_parity'] and r['output_sha256']==baseline[r['name']]['output_sha256'];events=r['events'];childcounts.update(e['kernel'] for e in events)
 if r['name'] in ('identity_self','dense_C_self','dense_F_self'):
  assert [e['m'] for e in events if e['kernel']=='scal']==[4,3,2,1];assert [(e['m'],e['n'],e['k']) for e in events if e['kernel']=='gemm']==[(4,4,4),(0,4,4)]
 elif r['name'] in ('strided_self','tall_transposed_right'):
  assert events[0]['kernel']=='small_permit' and not events[0]['result'];gemm=[e for e in events if e['kernel']=='gemm'];assert all(e['m']==4 and e['k']==4 for e in gemm);assert sum(e['n'] for e in gemm)==(4 if r['name']=='strided_self' else 1000)
 else:
  assert len(events)==2 and events[0]['kernel']=='small_permit' and events[0]['result'];e=events[1];assert e['kernel']==('small_b0_nt' if r['name'] in ('mixed_F_C','tall_F_left') else 'small_b0_nn');assert (e['m'],e['n'],e['k'])==(4,4 if r['name'] in ('independent_C','mixed_F_C') else 1000,4)
out=dict(cases=12,dispatch_counts=dict(counts),child_call_counts=dict(childcounts),exact_integer_products_reconstructed=True,strided_self_copy_changes_dispatch=True,review_cpu=time.process_time()-start,scope='Independent fixture bytes/layouts, integer products, passthrough parity and native call records checked. No mass/model replay or native kernel price; contiguous selftranspose observations do not establish all layout dispatch.')
Path(__file__).with_name('matmul_dispatch_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
