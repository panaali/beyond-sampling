"""Three fixed first-copy pilots; only after provenance and source gates."""
import time
START=time.process_time()
from pathlib import Path
import hashlib,importlib.util,json,resource,sys,traceback
HERE=Path(__file__).resolve().parent

def main():
 out=HERE/'pilots.json';progress=HERE/'pilots_progress.jsonl';assert not out.exists() and not progress.exists()
 gate=json.loads((HERE/'pilot_gate.json').read_text())
 for name,pin in gate['pins'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==pin,name
 provenance=json.loads((HERE/'provenance.json').read_text());assert provenance['complete'] and provenance['cases']==3840
 manifest=json.loads((HERE/'manifest.json').read_text())
 for name,pin in manifest['pins'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==pin,name
 native=Path(manifest['native_root']);sys.path.insert(0,str(native))
 spec=importlib.util.spec_from_file_location('t184_native_trace',native/'trace.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
 assert Path(T.B.__file__).resolve()==native/'base_snapshot.py'
 import numpy as np
 import torch
 assert T.np is np
 rows=[];result={'complete':False,'rows':rows,'gate':gate,'source_sha256':hashlib.sha256(T.SOURCE.read_bytes()).hexdigest(),
  'trace_sha256':hashlib.sha256((native/'trace.py').read_bytes()).hexdigest(),'numpy_version':np.__version__,'torch_version':torch.__version__,
  'scope':'Exactly H4/5/6 zoo0 noise80 copy0; original absolute .05 models. Direct named source arithmetic, not hardware/full-native cost or new accuracy panel.'}
 def save():
  result.update(cpu_seconds=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  out.write_text(json.dumps(result,indent=2)+'\n')
 for h in (4,5,6):
  row={'H':h,'zoo':0,'noise':80,'copy':0,'stage':'attempt'};rows.append(row);save()
  with progress.open('a') as f:f.write(json.dumps(row)+'\n')
  try:
   cp=next(c for c in manifest['checkpoints'] if (c['H'],c['zoo'])==(h,0));sd=torch.load(cp['path'],map_location='cpu',weights_only=True)
   base=(sd['rnn.weight_ih_l0'].numpy().astype(float)[:,0],sd['rnn.weight_hh_l0'].numpy().astype(float),sd['linear.weight'].numpy().astype(float))
   assert [w.shape for w in base]==[(h,),(h,h),(3,h)] and all(np.isfinite(w).all() for w in base)
   rng=np.random.default_rng(80);model=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
   sha=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest()
   old=json.loads((HERE/f'frozen/core/core_H{h}_z0_n80.json').read_text())['rows'][0];assert sha==old['model_sha256']
   row.update(model_sha256=sha,checkpoint_sha256=cp['sha256'],saved=old,stage='model_bound');save()
   baseline=T.evaluate(model,False);row.update(baseline=baseline,stage='baseline_done');save()
   traced=T.evaluate(model,True);row.update(traced=traced,stage='traced_done');save()
   for key in ('hex','legacy','stats','info'):assert baseline[key]==traced[key],(h,key)
   assert baseline['legacy']==old['saved_ops'] and baseline['stats']['cells']==old['saved_cells']
   assert baseline['value'].hex()==float(old['saved_estimate']).hex(),(h,'saved source bit parity')
   row.update(stage='complete',parity=True,named_numeric=traced['ledger']['named_numeric_total'],integer_control=traced['ledger']['integer_control_total'],sampler_numeric=4*h*h+12*h-1)
   save()
   with progress.open('a') as f:f.write(json.dumps(row)+'\n')
  except Exception:
   row.update(stage='failed',failure=traceback.format_exc());save()
   with progress.open('a') as f:f.write(json.dumps(row)+'\n')
   raise
 result.update(complete=True,model_reconstructions=3,source_estimator_calls=6,new_truth_or_MC_calls=0);save()
 print(json.dumps({'complete':True,'rows':[{'H':r['H'],'named_numeric':r['named_numeric'],'legacy':r['baseline']['legacy']} for r in rows],'cpu_seconds':result['cpu_seconds']}))
if __name__=='__main__':main()
