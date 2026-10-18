"""Load only the canonical pre-existing checkpoint; no model inputs/evaluations."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent

def weights():
 meta=json.loads((HERE/'checkpoint_identity.json').read_text());p=Path(meta['local_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
 assert meta['canonical_headers']['x-goog-hash']=='md5='+meta['md5_base64']
 d=torch.load(p,map_location='cpu',weights_only=True)
 d={k:v.detach().numpy().astype(float) for k,v in d.items()}
 model=(d['rnn.weight_ih_l0'][:,0].copy(),d['rnn.weight_hh_l0'].copy(),d['linear.weight'].copy())
 assert [w.shape for w in model]==[(4,),(4,4),(4,4)]
 assert all(np.all(np.isfinite(w)) for w in model)
 return model

def main():
 dst=HERE/'model_identity.json';assert not dst.exists();model=weights();out={'checkpoint_sha256':json.loads((HERE/'checkpoint_identity.json').read_text())['sha256'],'conversion':'Exact original zoo44: detach.numpy.astype(float), input column0.copy; other matrices copy','float64_c_order_sha256':hashlib.sha256(b''.join(w.astype('<f8').tobytes(order='C') for w in model)).hexdigest(),'model':[w.tolist() for w in model],'scope':'Identity only, no forward or estimator evaluation.'};dst.write_text(json.dumps(out,indent=2)+'\n');print(out['float64_c_order_sha256'])
if __name__=='__main__':main()
