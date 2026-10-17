"""Recover original first absolute-noise copy; no model input draws."""
from pathlib import Path
import hashlib,json,numpy as np,torch
from observer import HERE,load
FX,F3,Ops=load()
from zoo import noisy_abs,best_seed
path=Path('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_1.pth')
assert best_seed(8,4)==1
sd=torch.load(path,weights_only=True,map_location='cpu')
base=tuple(sd[k].detach().numpy().astype(float).copy() for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
base=(base[0][:,0].copy(),base[1],base[2]);model=noisy_abs(base,1,.05,0)[0]
rng=np.random.default_rng(0);manual=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
assert all(np.array_equal(a,b) for a,b in zip(model,manual))
def digest(m):return hashlib.sha256(b''.join(np.asarray(w,dtype='<f8').tobytes(order='C') for w in m)).hexdigest()
out={'name':'abs0.05_n0_0','H':8,'T':4,'seed':1,'n_seqs':2**30,'sigma':.05,'rng_seed':0,'copy':0,'order':['Wi','Wh','Wo'],'dtype':'little-endian float64 C-order concatenation','checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'base_sha256':digest(base),'model_sha256':digest(model),'shapes':[list(w.shape) for w in model],'model':[w.tolist() for w in model],'base':[w.tolist() for w in base],'truth_inventory':'No identical noise reference in T131/T145/T148/T109: clean seed0/1 only. final3 noise seed0 sigma .005 RNG7 differs. T113 is T3. Model digest must match before future reuse.','no_model_inputs':True}
(HERE/'noise_identity.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('base','model')})
