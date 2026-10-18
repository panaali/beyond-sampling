"""Frozen first64 seed1/noise20 source-native cost gate; existing control0 reused."""
import time
START=time.process_time()
import gzip,json,sys,traceback,resource,signal
from pathlib import Path
import numpy as np
from provenance import HERE,DATA,ROOT,load,mh,sha
sys.path.insert(0,str(ROOT/'recipe85b_native_cost'))
import native as N
def cap(sig,frame):raise RuntimeError('Preserved105CPU soft cap')
signal.signal(signal.SIGXCPU,cap);resource.setrlimit(resource.RLIMIT_CPU,(105,110))
def main():
 path=HERE/'pilot.json';archive=HERE/'pilot.jsonl.gz';assert not path.exists() and not archive.exists()
 pins=json.loads((HERE/'provenance.json').read_text());assert pins['complete'] and pins['rows']==4096
 for p,d in pins['pins'].items():assert sha(Path(p))==d
 old=json.loads((DATA/'k30_s1_noise20.json').read_text());control=json.loads((ROOT/'recipe85b_native_cost/trained_seed1.json').read_text());rng=np.random.default_rng(20);base=load(1)
 out=dict(complete=False,seed=1,noise=20,block=0,copies=64,rows=[],sampler_numeric=111,sampler_legacy=72,provenance_sha256=sha(HERE/'provenance.json'),driver_sha256=sha(Path(__file__)),scope='RECIPE85B only; named numeric source units, not full runtime')
 def save():
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(path)
 try:
  with gzip.open(archive,'wt',compresslevel=6) as log:
   for j in range(64):
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);saved=old['rows'][j];assert mh(model)==saved['model_hash']
    if j==0:
     assert mh(model)==control['model_sha256'];baseline=control['baseline'];traced=control['traced']
    else:baseline=N.evaluate(model,False);traced=N.evaluate(model,True)
    for key in ('hex','legacy','stats','info'):assert baseline[key]==traced[key],(j,key)
    previous=saved['recipes']['r85b'];assert baseline['value']==previous['value'] and baseline['legacy']==previous['native_ops'] and baseline['stats']['cells']==previous['cells']
    ledger=traced['ledger'];row=dict(copy=j,model_sha256=mh(model),truth=saved['saved_truth'],value=baseline['value'],hex=baseline['hex'],legacy=baseline['legacy'],cells=baseline['stats']['cells'],numeric=ledger['named_numeric_total'],integer=ledger['integer_control_total'],reused_control=j==0)
    out['rows'].append(row);log.write(json.dumps(dict(copy=j,model_sha256=mh(model),baseline=baseline,traced=traced))+'\n');log.flush();save()
    if time.process_time()-START>90:raise RuntimeError('Frozen90CPU pilot stop; preserve partial')
  out.update(complete=True,archive_sha256=sha(archive));save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(rows=len(out['rows']),cpu=out['cpu'],numeric=sum(r['numeric'] for r in out['rows']))),flush=True)
if __name__=='__main__':main()
