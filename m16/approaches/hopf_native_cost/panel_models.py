"""Exact original clean/noise provenance, with no forward/model evaluation."""
import ast,hashlib,json,types
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def model_hash(model):return hashlib.sha256(b''.join(w.astype('<f8').tobytes(order='C') for w in model)).hexdigest()
def models():
 meta=json.loads((HERE/'panel_checkpoint_identity.json').read_text())
 path=HERE/'noise_snapshot.py';tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='noisy_copies'];assert len(tree.body)==1
 mod=types.ModuleType('pinned_noise');mod.np=np;exec(compile(tree,str(path),'exec'),mod.__dict__)
 out=[]
 for row in meta['checkpoints']:
  p=Path(row['local_path']);assert digest(p)==row['sha256']
  sd=torch.load(p,map_location='cpu',weights_only=True);sd={k:v.detach().numpy().astype(float) for k,v in sd.items()}
  b=(sd['rnn.weight_ih_l0'][:,0].copy(),sd['rnn.weight_hh_l0'].copy(),sd['linear.weight'].copy())
  assert [w.shape for w in b]==[(4,),(4,4),(4,4)]
  for copy,m in enumerate([b]+mod.noisy_copies(b,4,.05,50)):
   assert all(np.all(np.isfinite(w)) for w in m)
   out.append((row['seed'],copy,m))
 assert len(out)==25
 return out

def main():
 dst=HERE/'panel_models.json';assert not dst.exists()
 rows=[{'seed':s,'copy':i,'model_sha256':model_hash(m),'model':[w.tolist() for w in m]} for s,i,m in models()]
 assert rows[0]['model_sha256']==json.loads((HERE/'model_identity.json').read_text())['float64_c_order_sha256']
 out={'claim':'304a1ddb','noise_source_sha256':digest(HERE/'noise_snapshot.py'),'checkpoint_identity_sha256':digest(HERE/'panel_checkpoint_identity.json'),'noise':'Exact original noisy_copies(base,4,.05,50), resetting RNG50 separately per seed, Wi then Wh then Wo per copy','rows':rows,'scope':'Reconstruct only existing fixed panel; no forward/estimator/MC evaluation.'}
 dst.write_text(json.dumps(out,indent=2)+'\n');print({'models':len(rows),'first_hash':rows[0]['model_sha256']})
if __name__=='__main__':main()
