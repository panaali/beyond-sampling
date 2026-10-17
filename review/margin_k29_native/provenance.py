"""Read-only complete saved model/noise provenance, without estimator calls."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
DATA=Path('/workspace/work/recover-small_h2_56/m16/approaches/training_axis_ladder')
ROOT=HERE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mh(m):return hashlib.sha256(b''.join(w.tobytes() for w in m)).hexdigest()
def load(seed):
 p=DATA/f'data/H4_T3_n2p29_seed{seed}.pth';sd=torch.load(p,weights_only=True,map_location='cpu')
 return (sd['rnn.weight_ih_l0'].double().numpy()[:,0],sd['rnn.weight_hh_l0'].double().numpy(),sd['linear.weight'].double().numpy())
def main():
 start=time.process_time();manifest=json.loads((DATA/'manifest.json').read_text());pins={};blocks=[]
 for path in [DATA/'manifest.json',DATA/'common.py',DATA/'block.py',ROOT/'recipe85b_native_cost/native.py',ROOT/'recipe85b_native_cost/inventory.json',ROOT/'recipe112_native_cost/trace.py',ROOT/'recipe112_native_cost/base_snapshot.py',ROOT/'recipe112_native_cost/inventory.json',ROOT/'recipe112_recovery/dependencies/understand_m43/est75.py']:
  if not path.exists():raise FileNotFoundError(path)
  pins[str(path)]=sha(path)
 for seed in range(5):
  path=DATA/f'data/H4_T3_n2p29_seed{seed}.pth';expected=next(r for r in manifest['checkpoints'] if r['power']==29 and r['seed']==seed);assert sha(path)==expected['sha256'];pins[str(path)]=sha(path);base=load(seed)
  for noise in range(20,24):
   path=DATA/f'block_s{seed}_n{noise}.json';old=json.loads(path.read_text());assert old['complete'] and len(old['rows'])==256 and old['base_sha256']==mh(base);rng=np.random.default_rng(noise)
   assert old['power']==29 and old['training_seed']==seed and old['noise_seed']==noise and old['sigma']==.05
   schema=old['schema']
   for j,row in enumerate(old['rows']):
    r=dict(zip(schema,row));model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);assert r['copy']==j and r['model_sha256']==mh(model)
   pins[str(path)]=sha(path);blocks.append(dict(seed=seed,noise=noise,rows=256,sha256=sha(path)))
 out=dict(complete=True,pins=pins,blocks=blocks,rows=5120,cpu=time.process_time()-start,driver_sha256=sha(Path(__file__)),truth_review='e49308c',claim='5be203e1')
 target=HERE/'provenance.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(rows=5120,cpu=out['cpu'])))
if __name__=='__main__':main()
