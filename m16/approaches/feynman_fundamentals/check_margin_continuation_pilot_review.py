import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery');start=time.process_time();manifest=json.loads((P/'continuation_manifest.json').read_text());d=json.loads((P/'continuation_pilot.json').read_text());assert d['complete'] and len(d['rows'])==73
for name,r in manifest.items():
 path=Path(r['source']) if name.startswith('checkpoint_') else P/'dependencies'/name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
rung=json.loads((P/'frozen/rung_k30.json').read_text())['seeds'];old=json.loads((P/'frozen/margin_k30.json').read_text())['seeds']['0'];expected={}
for seed in range(5):
 st=torch.load(manifest[f'checkpoint_{seed}']['source'],map_location='cpu',weights_only=True);base=tuple(x.numpy().astype(float) for x in [st['rnn.weight_ih_l0'][:,0],st['rnn.weight_hh_l0'],st['linear.weight']]);expected[seed,-1]=hashlib.sha256(b''.join(a.tobytes() for a in base)).hexdigest()
 for ns in range(20,24):
  rng=np.random.default_rng(ns)
  for j in range(1 if seed==0 else 4):
   m=tuple(w+.05*rng.standard_normal(w.shape) for w in base);expected[seed,(ns-20)*256+j]=hashlib.sha256(b''.join(a.tobytes() for a in m)).hexdigest()
seen=set();maxdiff=0
for r in d['rows']:
 key=(r['seed'],r['index']);assert key not in seen and r['model_hash']==expected[key];seen.add(key);truth=rung[str(r['seed'])]['clean']['truth'] if r['index']<0 else rung[str(r['seed'])]['panel']['truths'][r['index']];assert r['saved_truth']==truth and r['truth_difference']==r['independent_truth']-truth;maxdiff=max(maxdiff,abs(r['truth_difference']));assert abs(r['truth_difference'])<1e-11
 if r['seed']==0:
  for name,z in r['recipes'].items():
   target=old['clean'][name]['err'] if r['index']<0 else old['p20'][name]['errs'][r['index']];assert abs(z['value']-truth-target)<1e-12
   if r['index']<0:assert z['native_ops']==old['clean'][name]['ops'] and z['cells']==old['clean'][name]['cells']
assert seen==set(expected)
out=dict(pilot_rows=73,all_weight_hashes_reconstructed=True,dependency_hashes_verified=len(manifest),maximum_independent_truth_difference=maxdiff,review_cpu=time.process_time()-start,scope='Finite pilot provenance/source and saved parity, no estimator/truth rerun. Missing panels remain separate; historical72 and author-native counters are not complete common-unit costs.')
Path(__file__).with_name('margin_continuation_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
