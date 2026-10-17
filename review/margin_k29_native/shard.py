"""Complete frozen k29 panel in64-copy source-identical native shards."""
import time
START=time.process_time()
import gzip,json,sys,traceback,resource
from pathlib import Path
import numpy as np
from control import HERE,DATA,ROOT,load,mh,sha,evaluate

def main(seed,noise,block):
 assert seed in range(5) and noise in range(20,24) and block in range(4);assert (seed,noise,block)!=(0,20,0),'Reuse pilot instead'
 folder=HERE/'panel';folder.mkdir(exist_ok=True);stem=f's{seed}_n{noise}_b{block}';path=folder/f'{stem}.json';archive=folder/f'{stem}.jsonl.gz';assert not path.exists() and not archive.exists()
 pins=json.loads((HERE/'provenance.json').read_text());assert pins['complete']
 for p,digest in pins['pins'].items():assert sha(Path(p))==digest
 pilot=json.loads((HERE/'pilot.json').read_text());assert pilot['complete'];control=json.loads((HERE/'control_s4.json').read_text());assert control['complete']
 old=json.loads((DATA/f'block_s{seed}_n{noise}.json').read_text());rng=np.random.default_rng(noise);base=load(seed)
 out=dict(complete=False,seed=seed,noise=noise,block=block,copies=64,rows=[],sampler_numeric=111,sampler_legacy=72,provenance_sha256=sha(HERE/'provenance.json'),driver_sha256=sha(Path(__file__)),control_driver_sha256=sha(HERE/'control.py'),control4_sha256=sha(HERE/'control_s4.json'))
 def save():
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(out,indent=2)+'\n');temp.replace(path)
 try:
  with gzip.open(archive,'wt',compresslevel=6) as log:
   for j in range(64*(block+1)):
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
    if j<64*block:continue
    saved=dict(zip(old['schema'],old['rows'][j]));assert mh(model)==saved['model_sha256'];reuse=(seed,noise,j)==(4,23,255)
    raw=dict(copy=j,model_sha256=mh(model),recipes={});row=dict(copy=j,truth=saved['truth'],model_sha256=mh(model),recipes={},reused_control=reuse)
    for name in ('r85b','r112'):
     if reuse:baseline=control['recipes'][name]['baseline'];traced=control['recipes'][name]['traced'];assert mh(model)==control['model_sha256']
     else:baseline=evaluate(model,name,False);traced=evaluate(model,name,True)
     for key in ('hex','legacy','stats','info','recipe'):assert baseline[key]==traced[key],(j,name,key)
     ledger=traced['ledger'];row['recipes'][name]=dict(value=baseline['value'],legacy=baseline['legacy'],numeric=ledger['named_numeric_total'],integer=ledger['integer_control_total'],cells=baseline['stats']['cells'],hex=baseline['hex'])
     raw['recipes'][name]=dict(baseline=baseline,traced=traced)
    out['rows'].append(row);log.write(json.dumps(raw)+'\n');log.flush();save()
    if time.process_time()-START>90:raise RuntimeError('Frozen90CPU shard stop; preserve partial')
  out.update(complete=True,archive_sha256=sha(archive));save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(shard=stem,rows=len(out['rows']),cpu=out['cpu'],numeric={name:sum(r['recipes'][name]['numeric'] for r in out['rows']) for name in ('r85b','r112')})),flush=True)
if __name__=='__main__':main(*map(int,sys.argv[1:]))
