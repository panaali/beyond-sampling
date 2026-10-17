"""Fixed original k28 continuation, with full source gates and recorded CPU envelope."""
import time
START=time.process_time()
import gzip,json,sys,traceback,resource
from pathlib import Path
import numpy as np
from controls import HERE,DATA,sha,load,mh,evaluate
def main(seed,noise,block):
 assert seed in range(5) and noise in range(20,24) and block in range(4)
 assert (seed,noise,block)!=(0,20,0),'Reuse pilot once'
 gate=json.loads((HERE/'continuation_gate.json').read_text());assert gate['complete']
 for p,d in gate['pins'].items():assert sha(Path(p))==d
 folder=HERE/'panel';folder.mkdir(exist_ok=True);stem=f's{seed}_n{noise}_b{block}';path=folder/f'{stem}.json';archive=folder/f'{stem}.jsonl.gz';assert not path.exists() and not archive.exists()
 prior_paths=[HERE/'controls.json',HERE/'pilot.json']+sorted(folder.glob('*.json'));used=0.
 for p in prior_paths:
  previous=json.loads(p.read_text());assert previous['complete'] and 'failure' not in previous,('Prior incomplete phase',str(p));used+=previous['cpu']
 controls=json.loads((HERE/'controls.json').read_text());identities=json.loads((HERE/'reconstruction.json').read_text());assert controls['complete'] and identities['complete']
 control_map={(r['seed'],r['noise'],r['copy']):r for r in controls['rows'] if r['noise'] is not None};assert len(control_map)==10
 identity=identities['records'][seed];expected=identity['noises'][noise-20];assert identity['seed']==seed and expected['noise']==noise
 margin=json.loads((DATA/'frozen/margin_k28.json').read_text());rung=json.loads((DATA/'frozen/rung_k28.json').read_text());base=load(seed);assert mh(base)==identity['base_sha256'];rng=np.random.default_rng(noise)
 out=dict(started=True,complete=False,seed=seed,noise=noise,block=block,copies=64,rows=[],driver_sha256=sha(Path(__file__)),gate_sha256=sha(HERE/'continuation_gate.json'),controls_sha256=sha(HERE/'controls.json'),reconstruction_sha256=sha(HERE/'reconstruction.json'),prior_recorded_research_cpu=used,research_cpu_envelope=3000,sampler_numeric=111,sampler_legacy=72,historical_identity_verified=False,scope='Fixed original current-canonical k28 panel; saved-error/source parity and direct costs, no fullruntime or historical-byte proof.')
 def save():
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(path)
 save()
 try:
  assert used+110<=3000,'Recorded research envelope lacks a full process reserve; no new evaluator calls'
  with gzip.open(archive,'wt',compresslevel=6) as log:
   for j in range(64*(block+1)):
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
    if j<64*block:continue
    digest=mh(model);assert digest==expected['hashes'][j];truth=rung['seeds'][str(seed)]['panel']['truths'][(noise-20)*256+j];control=control_map.get((seed,noise,j));reuse=control is not None
    row=dict(copy=j,model_sha256=digest,truth=truth,reused_control=reuse,recipes={});out['rows'].append(row);save()
    for name in ('r85b','r112'):
     raw=dict(copy=j,model_sha256=digest,recipe=name,reused_control=reuse,stage='paired_before_validation')
     if reuse:
      assert digest==control['model_sha256'] and control['truth']==truth;baseline=control['recipes'][name]['baseline'];traced=control['recipes'][name]['traced']
     else:
      baseline=evaluate(model,name,False);log.write(json.dumps(dict(copy=j,recipe=name,stage='baseline',baseline=baseline))+'\n');log.flush()
      traced=evaluate(model,name,True)
     raw.update(baseline=baseline,traced=traced);log.write(json.dumps(raw)+'\n');log.flush()
     for key in ('hex','legacy','stats','info','recipe'):assert baseline[key]==traced[key],(seed,noise,j,name,key)
     old_error=margin['seeds'][str(seed)]['p20'][name]['errs'][(noise-20)*256+j];assert baseline['value']-truth==old_error,(seed,noise,j,name,'original subtraction mismatch')
     ledger=traced['ledger'];row['recipes'][name]=dict(value=baseline['value'],hex=baseline['hex'],old_error=old_error,legacy=baseline['legacy'],cells=baseline['stats']['cells'],numeric=ledger['named_numeric_total'],integer=ledger['integer_control_total']);save()
     if time.process_time()-START>90:raise RuntimeError('Frozen90CPU shard stop')
  out.update(complete=True,archive_sha256=sha(archive));save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(shard=stem,complete=True,models=len(out['rows']),cpu=out['cpu'],numeric={name:sum(r['recipes'][name]['numeric'] for r in out['rows']) for name in ('r85b','r112')})),flush=True)
if __name__=='__main__':main(*map(int,sys.argv[1:]))
