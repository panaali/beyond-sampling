"""Verify current checkpoint bytes and saved model reconstruction, no evaluations."""
import resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
start=time.process_time()
from pathlib import Path
import base64,hashlib,json
import numpy as np
import torch

K=Path('/workspace/work/recover-small_h4_4/review/margin_k28_native')
H=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_native')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mh(m):return hashlib.sha256(b''.join(w.astype('<f8').tobytes(order='C') for w in m)).hexdigest()
def load(p,h,t):
 s=torch.load(p,map_location='cpu',weights_only=True)
 shapes={'rnn.weight_ih_l0':(h,1),'rnn.weight_hh_l0':(h,h),'linear.weight':(t,h)}
 assert set(s)==set(shapes)
 for name,shape in shapes.items():assert tuple(s[name].shape)==shape and s[name].dtype==torch.float32
 a=[s[k].numpy().astype(np.float64) for k in shapes]
 m=(a[0][:,0].copy(),a[1],a[2]);assert all(np.isfinite(w).all() for w in m)
 return m
def payload(row,p):
 data=p.read_bytes();md5=base64.b64encode(hashlib.md5(data).digest()).decode()
 assert sha(p)==row['sha256'] and len(data)==row['bytes'] and md5==row['md5_base64']
 return md5
def noise(m,seed,count):
 rng=np.random.default_rng(seed)
 for _ in range(count):yield tuple(w+.05*rng.standard_normal(w.shape) for w in m)

kr=read(K/'reconstruction.json');km=read(K/'checkpoint_recovery_complete.json');failed=read(K/'checkpoint_recovery.json')
assert kr['complete'] and km['complete'] and not failed['complete']
for p,h in kr['pins'].items():assert sha(Path(p))==h
gate=read(K/'reconstruction_gate.json')
for p,h in gate['pins'].items():assert sha(Path(p))==h
assert km['prior_failure_sha256']==sha(K/'checkpoint_recovery.json')
assert km['historical_checkpoint_identity_verified'] is False and kr['historical_identity_verified'] is False
seen=set();kn=0
assert len(km['records'])==len(kr['records'])==5
for seed,(record,saved) in enumerate(zip(km['records'],kr['records'])):
 assert record['seed']==saved['seed']==seed and record['validated']
 p=K/'checkpoints'/f'k28_seed{seed}.pth';md5=payload(record,p)
 assert record['url']==f'https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_4_3_268435456_{seed}.pth'
 generation=record['head_headers']['x-goog-generation'];assert generation.isdigit()
 assert record['fetch_url']==record['url']+'?generation='+generation
 for header in (record['head_headers'],record['get_headers']):
  if header is None:assert seed==0;continue
  assert header['x-goog-generation']==generation and int(header['content-length'])==record['bytes']
  assert 'md5='+md5 in [x.strip() for x in header['x-goog-hash'].split(',')]
 assert record['payload_reused']==(seed==0)
 if seed==0:assert record['sha256']==failed['records'][0]['sha256']
 m=load(p,4,3)
 assert mh(m)==saved['base_sha256']==mh(tuple(np.asarray(w,dtype=np.float64) for w in saved['base']))
 assert [q['noise'] for q in saved['noises']]==list(range(20,24))
 for q in saved['noises']:
  hashes=[mh(n) for n in noise(m,q['noise'],256)]
  assert hashes==q['hashes'] and len(hashes)==256
  assert not seen.intersection(hashes);seen.update(hashes);kn+=len(hashes)
assert kn==kr['models']==len(seen)==5120

hr=read(H/'models.json');hm=read(H/'checkpoint_identity.json');hp=read(H/'protocol.json')
assert hr['complete'] and hm['complete'] and len(hr['models'])==25 and len(hm['checkpoints'])==5
assert hr['source_sha256']==sha(H/'provenance.py')
assert hr['checkpoint_identity_sha256']==sha(H/'checkpoint_identity.json')
assert hr['protocol_sha256']==hm['protocol_sha256']==sha(H/'protocol.json')
for mapping in (hp['source_pins'],hp['frozen_pins']):
 for p,h in mapping.items():assert sha(Path(p))==h
saved_report=read(H/'frozen/ev_54.json');hseen=set()
for seed,record in enumerate(hm['checkpoints']):
 p=H/'checkpoints'/f'seed{seed}.pth';md5=payload(record,p)
 assert record['seed']==seed and record['status']=='complete' and record['http_status']==200
 assert record['canonical_url']==hp['canonical_urls'][seed]
 headers=record['headers'];assert int(headers['content-length'][0])==record['bytes']
 assert headers['x-goog-generation'][0].isdigit() and 'md5='+md5 in headers['x-goog-hash']
 m=load(p,5,4)
 assert hr['checkpoints'][seed]['float64_hash']==mh(m)
 for copy,model in enumerate([m]+list(noise(m,50,4))):
  r=hr['models'][5*seed+copy];prior=saved_report[str(seed)]['rows'][copy]
  assert (r['seed'],r['copy'])==(seed,copy)
  digest=mh(model);assert digest==r['model_sha256'] and digest not in hseen;hseen.add(digest)
  assert all(a.tobytes()==np.asarray(b,dtype=np.float64).tobytes() for a,b in zip(model,r['model']))
  assert r['saved_h64']==prior['h64'] and r['saved_h64_hex']==float(prior['h64']).hex()
  assert r['saved_legacy']==prior['h64_ops'] and r['approximate_h128']==prior['ref']
  assert r['saved_mc']==prior['mc'] and r['mc_n']==10000000 and r['mc_rng']==(100+seed if copy==0 else 7)
  assert r['mc_count_inferred_from_report']==round(prior['mc']*10000000)
assert len(hseen)==25
out=dict(complete=True,k28_models=5120,h5_models=25,checkpoint_files_loaded=10,
 k28_reconstruction_sha256=sha(K/'reconstruction.json'),k28_recovery_sha256=sha(K/'checkpoint_recovery_complete.json'),
 h5_models_sha256=sha(H/'models.json'),h5_identity_sha256=sha(H/'checkpoint_identity.json'),
 review_cpu_seconds_including_imports=time.process_time()-start,review_noise_weight_arrays_generated=5140,
 evaluator_forward_reference_or_network_calls=0,
 scope='Current downloaded bytes, checksums/generations, exact state shapes/dtypes, original float64 conversion and all saved noisy-weight hashes independently checked. Historical checkpoint/per-copy identity remains unproved; saved-output/error parity is a separate future evaluator gate. H5 MC hits inferred from stored ratios only.')
Path(__file__).with_name('current_checkpoint_provenance_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
