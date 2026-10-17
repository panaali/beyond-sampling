"""Cheap first-mismatch Cantelli certificate usefulness falsifier."""
import importlib.util,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
start=time.process_time()
source=Path('/workspace/clean-neurons/m16/approaches/crossover_compute/estimator.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='68b0b5d7348a55b09a9cb80350a121823923517160bb364cd2b9fcf22bac2a2e'
spec=importlib.util.spec_from_file_location('table',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
means=[];covs=[]
for mean,cov in mod.moments(10,2):
 for top,second in itertools.permutations(range(10),2):
  order=[top,second]+[i for i in range(10) if i not in (top,second)];inv=np.argsort(order)
  means.append(mean[inv]);covs.append(cov[np.ix_(inv,inv)])
means=np.array(means);covs=np.array(covs);n=len(means)
path=Path('/workspace/work/recover-small_h2_56/m16/approaches/face_surplus_review/data/H16_T10_original.pth')
assert hashlib.sha256(path.read_bytes()).hexdigest()=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
st=torch.load(path,map_location='cpu',weights_only=True);base=[st[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];base[0]=base[0].ravel()
rows=[]
for sigma in (.0025,.02):
 rng=np.random.default_rng(1);wi,wh,wo=[v+sigma*np.sqrt(np.mean(v*v))*rng.standard_normal(v.shape) for v in base]
 jac=np.zeros((n,16,10));bounds=[];zero=0;minvar=0.
 for step in range(10):
  pre=np.einsum('ij,cjk->cik',wh,jac);pre[:,:,step]+=wi
  margin=np.einsum('cij,cj->ci',pre,means);variance=np.einsum('cij,cjk,cik->ci',pre,covs,pre)
  minvar=min(minvar,float(variance.min()));assert variance.min()>-1e-10
  variance=np.maximum(variance,0);identical=np.all(pre==0,axis=2);zero+=int(identical.sum())
  denominator=variance+margin**2
  bound=np.divide(variance,denominator,out=np.ones_like(variance),where=denominator>0);bound[identical]=0
  bounds.append(bound);jac=pre*(margin>0)[:,:,None]
 raw=np.sum(bounds,axis=(0,2));rows.append(dict(sigma=sigma,cells=n,gate_events=n*160,identically_zero_events=zero,variance_nonnegative_lower_check=minvar,raw_union_min=float(raw.min()),raw_union_mean=float(raw.mean()),raw_union_max=float(raw.max()),capped_average=float(np.minimum(raw,1).mean()),cells_with_bound_below_one=int(np.sum(raw<1)),average_individual_gate_bound=float(np.mean(bounds))))
out=dict(results=rows,cpu_seconds=time.process_time()-start,model_input_samples=0,scope='conditional real-arithmetic Cantelli bound using floating model-free moments; no interval certificate')
Path(__file__).with_name('gate_certificate_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
