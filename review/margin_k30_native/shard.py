"""Frozen original missing-k30 continuation. Gates and prior evidence are immutable."""
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
def main(seed,noise,block):
 assert seed in range(1,5) and noise in range(20,24) and block in range(4)
 assert (seed,noise,block)!=(1,20,0),'Reuse first64 pilot once'
 gate=json.loads((HERE/'pilot_review.json').read_text());assert gate['complete']
 for p,d in gate['pins'].items():assert sha(Path(p))==d
 folder=HERE/'panel';folder.mkdir(exist_ok=True);stem=f's{seed}_n{noise}_b{block}';path=folder/f'{stem}.json';archive=folder/f'{stem}.jsonl.gz';assert not path.exists() and not archive.exists()
 pins=json.loads((HERE/'provenance.json').read_text());assert pins['complete'] and pins['rows']==4096
 for p,d in pins['pins'].items():assert sha(Path(p))==d
 old=json.loads((DATA/f'k30_s{seed}_noise{noise}.json').read_text());control=json.loads((ROOT/'recipe85b_native_cost/trained_seed4.json').read_text());rng=np.random.default_rng(noise);base=load(seed)
 out=dict(complete=False,seed=seed,noise=noise,block=block,copies=64,rows=[],sampler_numeric=111,sampler_legacy=72,provenance_sha256=sha(HERE/'provenance.json'),driver_sha256=sha(Path(__file__)),pilot_gate_sha256=sha(HERE/'pilot_review.json'),scope='RECIPE85B only; named numeric source units, not full runtime')
 def save():
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(path)
 try:
  with gzip.open(archive,'wt',compresslevel=6) as log:
   for j in range(64*(block+1)):
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
    if j<64*block:continue
    saved=old['rows'][j];assert mh(model)==saved['model_hash'];reuse=(seed,noise,j)==(4,23,255)
    if reuse:
     assert mh(model)==control['model_sha256'];baseline=control['baseline'];traced=control['traced']
    else:baseline=N.evaluate(model,False);traced=N.evaluate(model,True)
    for key in ('hex','legacy','stats','info'):assert baseline[key]==traced[key],(j,key)
    previous=saved['recipes']['r85b'];assert baseline['value']==previous['value'] and baseline['legacy']==previous['native_ops'] and baseline['stats']['cells']==previous['cells']
    ledger=traced['ledger'];row=dict(copy=j,model_sha256=mh(model),truth=saved['saved_truth'],value=baseline['value'],hex=baseline['hex'],legacy=baseline['legacy'],cells=baseline['stats']['cells'],numeric=ledger['named_numeric_total'],integer=ledger['integer_control_total'],reused_control=reuse)
    out['rows'].append(row);log.write(json.dumps(dict(copy=j,model_sha256=mh(model),baseline=baseline,traced=traced))+'\n');log.flush();save()
    if time.process_time()-START>90:raise RuntimeError('Frozen90CPU shard stop; preserve partial')
  out.update(complete=True,archive_sha256=sha(archive));save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(shard=stem,rows=len(out['rows']),cpu=out['cpu'],numeric=sum(r['numeric'] for r in out['rows']))),flush=True)
if __name__=='__main__':main(*map(int,sys.argv[1:]))
