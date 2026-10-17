"""Independent unconditional sampler and direct frozen-mask recurrence."""
import time,sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr,betainc
start=time.process_time()
source=Path('/workspace/clean-neurons/m16/approaches/crossover_compute/estimator.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='68b0b5d7348a55b09a9cb80350a121823923517160bb364cd2b9fcf22bac2a2e'
spec=importlib.util.spec_from_file_location('mom',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
table=module.moments(10,2)
weights=Path('/workspace/work/recover-small_h2_56/m16/approaches/face_surplus_review/data/H16_T10_original.pth')
assert hashlib.sha256(weights.read_bytes()).hexdigest()=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
state=torch.load(weights,map_location='cpu',weights_only=True)
base=[state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];base[0]=base[0].ravel()
noise=np.random.default_rng(1);wi,wh,wo=[v+.02*np.sqrt(np.mean(v*v))*noise.standard_normal(v.shape) for v in base]
x=np.random.default_rng(28103).standard_normal((8192,10));n=len(x);order=np.argsort(x,axis=1);maximum,second=order[:,-1],order[:,-2]
q=ndtr(x[np.arange(n),second]);stratum=(betainc(9,2,q)>=.5).astype(int)
mu=np.empty_like(x)
for k in (0,1):
 rows=np.flatnonzero(stratum==k);m=table[k][0]
 mu[rows]=m[2];mu[rows,maximum[rows]]=m[0];mu[rows,second[rows]]=m[1]
hmean=np.zeros((n,16));htrue=hmean.copy();hfrozen=hmean.copy();hrescue=None
for t in range(10):
 hmean=np.maximum(hmean@wh.T+mu[:,t,None]*wi,0);gate=hmean>0
 htrue=np.maximum(htrue@wh.T+x[:,t,None]*wi,0)
 hfrozen=(hfrozen@wh.T+x[:,t,None]*wi)*gate
 if t==7:hrescue=hfrozen.copy()
 elif t>7:hrescue=np.maximum(hrescue@wh.T+x[:,t,None]*wi,0)
truth=(htrue@wo.T).argmax(axis=1)==second
rows=[]
for name,h in [('frozen',hfrozen),('last2_exact',hrescue)]:
 correct=(h@wo.T).argmax(axis=1)==second;d=correct.astype(float)-truth
 rows.append(dict(method=name,accuracy=float(correct.mean()),paired_bias=float(d.mean()),paired_se=float(d.std(ddof=1)/np.sqrt(n)),disagreement=float(np.mean(correct!=truth))))
out=dict(seed=28103,n=n,sigma=.02,actual_accuracy=float(truth.mean()),stratum_counts=np.bincount(stratum).tolist(),results=rows,cpu_seconds=time.process_time()-start,normal_input_variates=n*10,hidden_step_vectors=n*32,method='diagnostic iid input sampling with independent direct fixed-mask recurrence')
Path(__file__).with_name('rank_gate_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
