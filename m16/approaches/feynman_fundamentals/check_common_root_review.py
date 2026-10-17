import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
start=time.process_time();source=Path('/workspace/work/recover-small_h2_56/m16/approaches/compiled_geometry/mixed_diagnostic.json');old=json.loads(source.read_text())
g=np.random.default_rng(660001).standard_normal((32,10,2));q,r=np.linalg.qr(g);q*=np.sign(np.diagonal(r,axis1=1,axis2=2))[:,None,:]
u,v=q[:,:,0],q[:,:,1]
paths=[Path('/workspace/work/recover-small_h2_56/m16/approaches/face_surplus_review/data/H16_T10_original.pth'),Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth')]
models=[]
for path in paths:
 st=torch.load(path,map_location='cpu',weights_only=True);model=[st[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];model[0]=model[0].ravel();models.append(model)
cases=[]
for sigma in (0,.0025,.02):
 rng=np.random.default_rng(1);cases.append(models[0] if sigma==0 else [w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in models[0]])
cases.append(models[1]);rows=[]
for model,saved in zip(cases,old['cases']):
 w=model[0];root_counts=[]
 for i in range(32):
  a=w*u[i,0];b=w*v[i,0];mixed=-a.astype(np.longdouble)/b.astype(np.longdouble)
  a=w.astype(np.longdouble)*np.longdouble(u[i,0]);b=w.astype(np.longdouble)*np.longdouble(v[i,0]);stable=-a/b
  nc,ns=len(np.unique(mixed)),len(np.unique(stable));root_counts.append((nc,ns))
 predicted=[2*(nc-ns) for nc,ns in root_counts];actual=saved['piece_difference']
 rows.append(dict(name=saved['name'],mixed_stable_distinct_first_roots=root_counts,predicted_extra_pieces=predicted,observed_extra_pieces=actual,exact_match_planes=sum(a==b for a,b in zip(predicted,actual)),residual=[a-b for a,b in zip(actual,predicted)]))
out=dict(cases=rows,cpu_seconds=time.process_time()-start,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_forward_calls=0)
Path(__file__).with_name('common_root_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{k:v for k,v in row.items() if k!='mixed_stable_distinct_first_roots'} for row in rows],indent=2))
