"""Frozen first64 seed0/noise20 native-cost gate; reuse control0 once."""
import time
START=time.process_time()
import gzip,json,sys,traceback,resource
import numpy as np
from control import HERE,DATA,ROOT,load,mh,sha,evaluate

def main():
 path=HERE/'pilot.json';archive=HERE/'pilot.jsonl.gz';assert not path.exists() and not archive.exists()
 pins=json.loads((HERE/'provenance.json').read_text());assert pins['complete']
 for p,digest in pins['pins'].items():assert sha(__import__('pathlib').Path(p))==digest
 controls=[json.loads((HERE/f'control_s{s}.json').read_text()) for s in (0,4)];assert all(c['complete'] for c in controls)
 old=json.loads((DATA/'block_s0_n20.json').read_text());rng=np.random.default_rng(20);base=load(0)
 out=dict(complete=False,seed=0,noise=20,copies=64,rows=[],sampler_numeric=111,sampler_legacy=72,provenance_sha256=sha(HERE/'provenance.json'),driver_sha256=sha(__import__('pathlib').Path(__file__)),controls={str(s):sha(HERE/f'control_s{s}.json') for s in (0,4)})
 def save():
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);path.write_text(json.dumps(out,indent=2)+'\n')
 try:
  with gzip.open(archive,'wt',compresslevel=6) as log:
   for j in range(64):
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);saved=dict(zip(old['schema'],old['rows'][j]));assert mh(model)==saved['model_sha256'];raw=dict(copy=j,model_sha256=mh(model),recipes={})
    row=dict(copy=j,truth=saved['truth'],model_sha256=mh(model),recipes={},reused_control=j==0)
    for name in ('r85b','r112'):
     if j==0:baseline=controls[0]['recipes'][name]['baseline'];traced=controls[0]['recipes'][name]['traced'];assert mh(model)==controls[0]['model_sha256']
     else:baseline=evaluate(model,name,False);traced=evaluate(model,name,True)
     for key in ('hex','legacy','stats','info','recipe'):assert baseline[key]==traced[key],(j,name,key)
     ledger=traced['ledger'];row['recipes'][name]=dict(value=baseline['value'],legacy=baseline['legacy'],numeric=ledger['named_numeric_total'],integer=ledger['integer_control_total'],cells=baseline['stats']['cells'],hex=baseline['hex'])
     raw['recipes'][name]=dict(baseline=baseline,traced=traced)
    out['rows'].append(row);log.write(json.dumps(raw)+'\n');log.flush();save()
    if time.process_time()-START>90:raise RuntimeError('Frozen90CPU pilot phase stop; preserve partial')
  out.update(complete=True,archive_sha256=sha(archive));save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(rows=len(out['rows']),cpu=out['cpu'],numeric={name:sum(r['recipes'][name]['numeric'] for r in out['rows']) for name in ('r85b','r112')})),flush=True)
if __name__=='__main__':main()
