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
out=dict(cases=12,dispatch_counts=dict(counts),exact_integer_products_reconstructed=True,strided_self_copy_changes_dispatch=True,review_cpu=time.process_time()-start,scope='Independent fixture bytes/layouts, integer products, passthrough parity and native call records checked. No mass/model replay or native kernel price; contiguous selftranspose observations do not establish all layout dispatch.')
Path(__file__).with_name('matmul_dispatch_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
