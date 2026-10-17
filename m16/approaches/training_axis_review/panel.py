"""Execute a frozen T114 phase, checkpointing every model; never select by outcome."""
import json,time,sys,hashlib
from pathlib import Path
import numpy as np
import loaders as L
from reference import exact
HERE=Path(__file__).parent

def save(path,data):path.write_text(json.dumps(data,indent=2,default=lambda x:x.item())+'\n')
def score(m,do_fibre):
 t=time.process_time();p,n=exact(m);p_author=float(L.author_reference.exact_acc(m));a=L.t75(m)
 r=dict(truth=p,truth_leaves=n,author_truth=p_author,reference_difference=p-p_author,t75=a)
 r['checks']=dict(reference_agrees=abs(p-p_author)<1e-10,recipe_agrees=abs(a['value']-a['own_value'])<1e-10,probability_valid=0<=p<=1)
 if do_fibre:r['fibre']=L.fibre(m)
 r['cpu']=time.process_time()-t
 return r

def run_cell(k,s,kind):
 path=HERE/f'{kind}_k{k}_s{s}.json'
 base=L.load(k,s);sigma=.05 if kind=='absolute' else .05*L.rms(base)/L.rms(L.load(30,s))
 ns=(120,121) if kind=='absolute' else (122,123)
 n=128 if kind=='absolute' and k==24 else 32
 start=time.process_time();wall=time.perf_counter()
 d=dict(k=k,training_seed=s,kind=kind,sigma=sigma,noise_seeds=ns,n_per_seed=n,rows=[],complete=False)
 if path.exists():
  prior=json.loads(path.read_text())
  if prior.get('complete'):print(path.name,'already complete',flush=True);return
  # Incomplete outputs remain for audit; deterministic rerun records prior elapsed cost.
  d['previous_attempt']=prior
 try:
  d['clean']=score(base,True);save(path,d)
  for seed in ns:
   for index,m in enumerate(L.noisy(base,seed,n,sigma)):
    r=score(m,index<8);r.update(noise_seed=seed,index=index);d['rows'].append(r)
    d['cpu']=time.process_time()-start;d['wall']=time.perf_counter()-wall;save(path,d)
  d['complete']=True;save(path,d)
 except Exception as e:
  d['failure']=dict(type=type(e).__name__,message=str(e));d['cpu']=time.process_time()-start;save(path,d);raise
 print(path.name,'complete',len(d['rows']),'CPU',round(d['cpu'],3),'max truth difference',max(abs(z['reference_difference']) for z in d['rows']),flush=True)
if __name__=='__main__':
 arg=sys.argv[1]
 if arg=='absolute':
  run_cell(24,4,'absolute')
  for s in range(5):run_cell(20,s,'absolute')
 else:
  for s in range(5):run_cell(int(arg),s,'relative')
