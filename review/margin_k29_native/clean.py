"""Five frozen clean checkpoints, using existing independently checked truths."""
import time
START=time.process_time()
import json,resource,traceback
from pathlib import Path
from control import HERE,load,mh,sha,evaluate

def main():
 path=HERE/'clean.json';assert not path.exists();pins=json.loads((HERE/'provenance.json').read_text())
 for p,digest in pins['pins'].items():assert sha(Path(p))==digest
 pin=json.loads((HERE/'clean_pin.json').read_text());assert sha(Path(pin['source']))==pin['sha256']
 out=dict(complete=False,rows=[],source_sha256=sha(Path(__file__)),pin_sha256=sha(HERE/'clean_pin.json'),sampler_numeric=111,sampler_legacy=72)
 try:
  for seed,saved in enumerate(pin['rows']):
   model=load(seed);assert saved['training_seed']==seed and saved['model_sha256']==mh(model)
   row=dict(seed=seed,model_sha256=mh(model),truth=saved['truth']['value'],independent_truth=saved['independent']['value'],recipes={})
   for name in ('r85b','r112'):
    baseline=evaluate(model,name,False);traced=evaluate(model,name,True)
    for key in ('hex','legacy','stats','info','recipe'):assert baseline[key]==traced[key],(seed,name,key)
    row['recipes'][name]=dict(baseline=baseline,traced=traced)
   out['rows'].append(row)
  out['complete']=True
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());raise
 finally:out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);path.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(rows=len(out['rows']),cpu=out['cpu'])))
if __name__=='__main__':main()
