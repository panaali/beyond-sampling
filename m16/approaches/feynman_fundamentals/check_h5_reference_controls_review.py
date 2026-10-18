"""Saved forward-control/pin audit; no forward or random draws."""
from pathlib import Path
import hashlib,json,zlib,struct,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_reference');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((P/'protocol.json').read_text());control=json.loads((P/'controls.json').read_text())
for r in protocol['files'].values():assert sha(Path(r['path']))==r['sha256']
for p,h in json.loads((P/'source_scope.json').read_text())['pins'].items():assert sha(Path(p))==h
assert control['source_sha256']==sha(P/'reference.py') and control['protocol_sha256']==sha(P/'protocol.json')
assert control['complete'] and control['random_draws']==0 and len(control['rows'])==6
models=json.loads(Path(protocol['files']['models']['path']).read_text())['models']
oldh4=json.loads(Path(protocol['files']['h4_protocol']['path']).read_text());excluded={7,100,101,102,103,104}|{r['rng_seed'] for r in oldh4['models']}
seeds=[]
for i,(r,m) in enumerate(zip(protocol['models'],models,strict=True)):
 assert r['index']==i and (r['seed'],r['copy'])==divmod(i,5)
 assert r['rng_seed']==zlib.crc32(r['rng_name'].encode('ascii')) and r['rng_seed'] not in excluded;seeds.append(r['rng_seed'])
 data=b''.join(struct.pack('<d',x) for a in m['model'] for row in (a if isinstance(a[0],list) else [a]) for x in row)
 assert hashlib.sha256(data).hexdigest()==r['model_sha256']==m['model_sha256']
assert len(set(seeds))==len(models)==25 and control['model_sha256']==models[0]['model_sha256']
for r,(n,chunk) in zip(control['rows'],[(n,c) for n in (5,17,31) for c in (3,11)],strict=True):
 assert r['n']==n and r['chunk']==chunk and r['status']=='complete' and r['consumed']==[n,n]
 assert type(r['hits']) is int and r['hits']==r['direct_hits'] and 0<=r['hits']<=n
 assert float(r['hits']/n).hex()==r['value_hex']==float(r['value']).hex()
 x=[[(4*i+j)%19/4-9/4 for j in range(4)] for i in range(n)];x[0]=[0.]*4;x[1]=[1.]*4;x[2]=[1.,1.,0.,-1.]
 assert hashlib.sha256(b''.join(struct.pack('<d',v) for row in x for v in row)).hexdigest()==r['input_sha256']
assert sum(3*r['n'] for r in control['rows'])==control['completed_extra_forward_evaluations']==318
out={'pass':True,'cases':6,'extra_forwards':318,'random_draws':0,'model_pins':25,'control_sha256':sha(P/'controls.json'),'source_sha256':sha(P/'reference.py'),'protocol_sha256':sha(P/'protocol.json'),'author_cpu':control['cpu_seconds'],'review_cpu':time.process_time()-start,'scope':'Saved integer parity, deterministic input bytes, model/source pins and stream identity audit. No forwards or random draws. First random block still needs completed-control pilot scope.'}
(O/'h5_reference_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
