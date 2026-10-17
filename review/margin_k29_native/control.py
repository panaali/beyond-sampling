"""Two frozen baseline/native controls. No estimator or tracer edits."""
import time
START=time.process_time()
import hashlib,json,sys,resource,signal,traceback
from pathlib import Path
import numpy as np
from provenance import HERE,DATA,ROOT,load,mh,sha
sys.path.insert(0,str(ROOT/'recipe85b_native_cost'))
import native as N

def cap(sig,frame):raise RuntimeError('Preserved105CPU cap')
signal.signal(signal.SIGXCPU,cap);resource.setrlimit(resource.RLIMIT_CPU,(105,110))
def evaluate(model,name,instrument):
 # The reviewed T176 module uses the same T172 loader, adding tuple-sort accounting.
 mod,ledger,startup=N.N.load(instrument);mod.STATS={'cut':0,'resolved':0,'cells':0};ops=mod.Ops();info={}
 recipe=mod.RECIPE85B if name=='r85b' else mod.RECIPE112
 value=mod.est75(model,ops,info=info,**recipe)
 return dict(value=float(value),hex=float(value).hex(),legacy=ops.n,stats=mod.STATS,info=[dict(key=list(k),value=float(v)) for k,v in info['err_by'].items()],ledger=ledger.report(),startup=startup,recipe=recipe)
def run(seed):
 assert seed in (0,4);noise,copy=(20,0) if seed==0 else (23,255);target=HERE/f'control_s{seed}.json';assert not target.exists()
 pins=json.loads((HERE/'provenance.json').read_text());assert pins['complete']
 for path,digest in pins['pins'].items():assert sha(Path(path))==digest
 base=load(seed);rng=np.random.default_rng(noise)
 for _ in range(copy+1):model=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
 old=json.loads((DATA/f'block_s{seed}_n{noise}.json').read_text());saved=dict(zip(old['schema'],old['rows'][copy]));assert mh(model)==saved['model_sha256']
 out=dict(seed=seed,noise=noise,copy=copy,model_sha256=mh(model),saved_truth=saved['truth'],prior=saved,recipes={},complete=False,provenance_sha256=sha(HERE/'provenance.json'),driver_sha256=sha(Path(__file__)),sampler_numeric=111,sampler_legacy=72)
 try:
  for name in ('r85b','r112'):
   baseline=evaluate(model,name,False);traced=evaluate(model,name,True)
   for key in ('hex','legacy','stats','info','recipe'):assert baseline[key]==traced[key],(name,key)
   out['recipes'][name]=dict(baseline=baseline,traced=traced)
  out['complete']=True
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);target.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(seed=seed,cpu=out['cpu'],recipes={k:dict(value=v['baseline']['value'],numeric=v['traced']['ledger']['named_numeric_total']) for k,v in out['recipes'].items()})))
if __name__=='__main__':run(int(sys.argv[1]))
