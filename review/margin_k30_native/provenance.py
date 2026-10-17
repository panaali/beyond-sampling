"""Pin and reconstruct the existing missing-k30 panel, without estimator calls."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
DATA=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery')
ROOT=Path('/workspace/work/recover-small_h2_910/review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mh(m):return hashlib.sha256(b''.join(w.tobytes() for w in m)).hexdigest()
def load(seed):
 record=json.loads((DATA/'continuation_manifest.json').read_text())[f'checkpoint_{seed}'];p=Path(record['source']);assert sha(p)==record['sha256']
 sd=torch.load(p,weights_only=True,map_location='cpu')
 return (sd['rnn.weight_ih_l0'].double().numpy()[:,0],sd['rnn.weight_hh_l0'].double().numpy(),sd['linear.weight'].double().numpy())
def main():
 start=time.process_time();target=HERE/'provenance.json';assert not target.exists()
 inventory=json.loads((ROOT/'recipe85b_native_cost/inventory.json').read_text());pins=dict(inventory['pins'])
 for p,d in pins.items():assert sha(Path(p))==d
 for p in [Path(__file__),DATA/'continuation_summary.json',ROOT/'recipe85b_native_cost/native.py',ROOT/'recipe85b_native_cost/inventory.json',ROOT/'recipe112_recovery/dependencies.json']+[ROOT/f'recipe85b_native_cost/trained_seed{s}.json' for s in (1,4)]:pins[str(p)]=sha(p)
 blocks=[];hashes=set()
 for seed in range(1,5):
  record=json.loads((DATA/'continuation_manifest.json').read_text())[f'checkpoint_{seed}'];pins[record['source']]=record['sha256'];base=load(seed)
  for noise in range(20,24):
   p=DATA/f'k30_s{seed}_noise{noise}.json';old=json.loads(p.read_text());assert old['complete'] and old['seed']==seed and old['noise_seed']==noise and old['sigma']==.05 and len(old['rows'])==256
   rng=np.random.default_rng(noise)
   for j,r in enumerate(old['rows']):
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);digest=mh(model)
    assert r['seed']==seed and r['index']==(noise-20)*256+j and r['model_hash']==digest and digest not in hashes
    assert np.isfinite(r['saved_truth']) and 0<=r['saved_truth']<=1
    hashes.add(digest)
   pins[str(p)]=sha(p);blocks.append(dict(seed=seed,noise=noise,rows=256,sha256=sha(p)))
 controls=[]
 for seed,noise,j in [(1,20,0),(4,23,255)]:
  p=ROOT/f'recipe85b_native_cost/trained_seed{seed}.json';r=json.loads(p.read_text());saved=json.loads((DATA/f'k30_s{seed}_noise{noise}.json').read_text())['rows'][j]
  assert (r['seed'],r['noise'],r['copy'])==(seed,noise,j) and r['model_sha256']==saved['model_hash']
  for key in ('hex','legacy','stats','info'):assert r['baseline'][key]==r['traced'][key]
  assert r['baseline']['value']==saved['recipes']['r85b']['value'] and r['baseline']['legacy']==saved['recipes']['r85b']['native_ops']
  controls.append(dict(seed=seed,noise=noise,copy=j,sha256=sha(p),numeric=r['traced']['ledger']['named_numeric_total']))
 out=dict(complete=True,claim='9235ef05',rows=len(hashes),blocks=blocks,controls=controls,pins=pins,cpu=time.process_time()-start,truth_review='T115 missing-k30 7ec65db8; independent saved review',sampler_numeric=111,sampler_legacy=72)
 target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(rows=out['rows'],cpu=out['cpu'])))
if __name__=='__main__':main()
