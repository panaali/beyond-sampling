"""Current canonical weights and original noisy-copy ordering; no evaluator calls."""
import ast,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from inventory import HERE,DATA,sha
def mh(model):return hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest()
def load(seed):
 meta=json.loads((HERE/'checkpoint_recovery.json').read_text());assert meta['complete'];r=meta['records'][seed];assert r['seed']==seed and r['validated'];p=Path(r['path']);assert sha(p)==r['sha256']
 state=torch.load(p,map_location='cpu',weights_only=True)
 model=(state['rnn.weight_ih_l0'].double().numpy()[:,0].copy(),state['rnn.weight_hh_l0'].double().numpy().copy(),state['linear.weight'].double().numpy().copy())
 assert [w.shape for w in model]==[(4,),(4,4),(3,4)] and all(w.dtype==np.float64 and np.isfinite(w).all() for w in model)
 return model
def main():
 start=time.process_time();target=HERE/'reconstruction.json';assert not target.exists();inventory=json.loads((HERE/'inventory.json').read_text())
 for p,d in inventory['pins'].items():assert sha(Path(p))==d
 source=DATA/'dependencies/zoo43.py';tree=ast.parse(source.read_text());func=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='noisy_copies'];assert len(func)==1;namespace={'np':np};exec(compile(ast.Module(body=func,type_ignores=[]),str(source),'exec'),namespace)
 records=[];seen=set()
 for seed in range(5):
  model=load(seed);noises=[]
  for noise in range(20,24):
   original=namespace['noisy_copies'](model,256,.05,noise);rng=np.random.default_rng(noise);hashes=[]
   for j,reference in enumerate(original):
    current=tuple(w+.05*rng.standard_normal(w.shape) for w in model);assert all(a.tobytes()==b.tobytes() for a,b in zip(current,reference))
    digest=mh(current);assert digest not in seen;seen.add(digest);hashes.append(digest)
   noises.append(dict(noise=noise,hashes=hashes))
  records.append(dict(seed=seed,base_sha256=mh(model),base=[w.tolist() for w in model],noises=noises))
 out=dict(complete=True,models=len(seen),records=records,pins={str(p):sha(p) for p in (HERE/'inventory.json',HERE/'checkpoint_recovery.json',source,Path(__file__))},historical_identity_verified=False,cpu=time.process_time()-start,scope='All5120 current canonical noisy arrays match original generator byte-for-byte, newly recorded hashes only. No comparison against absent historical model hashes and no evaluator/truth calls.')
 target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('complete','models','historical_identity_verified','cpu')}))
if __name__=='__main__':main()
