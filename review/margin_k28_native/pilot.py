"""First64 seed0/noise20, both frozen recipes; fixed control enters once."""
import time
START=time.process_time()
import gzip,json,traceback,resource
from pathlib import Path
import numpy as np
from controls import HERE,DATA,sha,load,mh,evaluate
def main():
 path=HERE/'pilot.json';archive=HERE/'pilot.jsonl.gz';assert not path.exists() and not archive.exists()
 gate=json.loads((HERE/'pilot_gate.json').read_text());assert gate['complete']
 for p,d in gate['pins'].items():assert sha(Path(p))==d
 controls=json.loads((HERE/'controls.json').read_text());assert controls['complete']
 selected=[r for r in controls['rows'] if (r['seed'],r['noise'],r['copy'])==(0,20,0)];assert len(selected)==1;control=selected[0]
 identities=json.loads((HERE/'reconstruction.json').read_text());assert identities['complete'];expected=identities['records'][0]['noises'][0];assert expected['noise']==20
 margin=json.loads((DATA/'frozen/margin_k28.json').read_text());rung=json.loads((DATA/'frozen/rung_k28.json').read_text());base=load(0);rng=np.random.default_rng(20)
 out=dict(started=True,complete=False,seed=0,noise=20,block=0,copies=64,rows=[],driver_sha256=sha(Path(__file__)),gate_sha256=sha(HERE/'pilot_gate.json'),controls_sha256=sha(HERE/'controls.json'),reconstruction_sha256=sha(HERE/'reconstruction.json'),sampler_numeric=111,sampler_legacy=72,historical_identity_verified=False,scope='First64 current canonical/original-error cost pilot only; no full panel or actual-runtime claim.')
 def save():
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(path)
 save()
 try:
  with gzip.open(archive,'wt',compresslevel=6) as log:
   for j in range(64):
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);digest=mh(model);assert digest==expected['hashes'][j];truth=rung['seeds']['0']['panel']['truths'][j]
    row=dict(copy=j,model_sha256=digest,truth=truth,reused_control=j==0,recipes={});out['rows'].append(row);save()
    for name in ('r85b','r112'):
     raw=dict(copy=j,model_sha256=digest,recipe=name,reused_control=j==0,stage='paired_before_validation')
     if j==0:
      assert digest==control['model_sha256'] and control['truth']==truth;baseline=control['recipes'][name]['baseline'];traced=control['recipes'][name]['traced']
     else:
      baseline=evaluate(model,name,False);log.write(json.dumps(dict(copy=j,recipe=name,stage='baseline',baseline=baseline))+'\n');log.flush()
      traced=evaluate(model,name,True)
     raw.update(baseline=baseline,traced=traced)
     # Save both numerical arms before enforcing the comparison.
     log.write(json.dumps(raw)+'\n');log.flush()
     for key in ('hex','legacy','stats','info','recipe'):assert baseline[key]==traced[key],(j,name,key)
     old_error=margin['seeds']['0']['p20'][name]['errs'][j];assert baseline['value']-truth==old_error,(j,name,'original subtraction mismatch')
     ledger=traced['ledger'];row['recipes'][name]=dict(value=baseline['value'],hex=baseline['hex'],old_error=old_error,legacy=baseline['legacy'],cells=baseline['stats']['cells'],numeric=ledger['named_numeric_total'],integer=ledger['integer_control_total']);save()
     if time.process_time()-START>90:raise RuntimeError('Frozen90CPU pilot stop')
  out.update(complete=True,archive_sha256=sha(archive));save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(complete=True,models=len(out['rows']),cpu=out['cpu'],numeric={name:sum(r['recipes'][name]['numeric'] for r in out['rows']) for name in ('r85b','r112')})))
if __name__=='__main__':main()
