"""New model-facing capture vs saved arrays; never rerun trained mass integration."""
import time
START=time.process_time()
from pathlib import Path
import argparse,hashlib,json,math,resource,traceback
import numpy as np
import torch
import adapter
P=Path(__file__).resolve().parent

def main(seed):
 resource.setrlimit(resource.RLIMIT_CPU,(110,112));torch.set_num_threads(1)
 protocol=json.loads((P/'capture_protocol.json').read_text())
 for f,h in protocol['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 case=next(c for c in protocol['cases'] if c['seed']==seed)
 out=dict(seed=seed,complete=False,random_network_inputs=0,trained_mass_integrations=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((P/'capture_protocol.json').read_bytes()).hexdigest())
 try:
  E,guard,M,old,kernel,rescue,radial=adapter.load();Ops=E.Ops
  class BoundedOps(Ops):
   def add(self,n):
    super().add(n)
    if time.process_time()-START>95:raise RuntimeError('95CPU soft cap: incomplete capture')
  E.Ops=BoundedOps
  state=torch.load(case['weights'],map_location='cpu',weights_only=True)
  model=(state['rnn.weight_ih_l0'].numpy().astype(float)[:,0],state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float));assert model[1].shape==(8,8) and model[2].shape==(4,8)
  # No reference arrays or mass values are opened until model collection finishes.
  rays,capture=adapter.collect(model,E,guard);out['capture']=capture
  ref=np.load(case['reference_capture']);metadata=json.loads(Path(case['capture_metadata']).read_text());assert len(rays)==len(ref['rays'])
  mismatch=[i for i,R in enumerate(rays) if not np.array_equal(R,ref['rays'][i])];assert not mismatch,mismatch[:10]
  digest=hashlib.sha256()
  for R in rays:digest.update(R.tobytes())
  assert capture['stats']==metadata['stats']
  assert capture['guard_counts'].get('earlier_tie',0)==capture['guard_counts'].get('later_tie',0)==0
  assert capture['guard_counts'].get('integer_orientation_comparisons',0)==0
  prior=np.load(case['positive_values']);summary=json.loads(Path(case['positive_summary']).read_text());assert len(prior['values'])==len(rays)
  saved_accuracy=1-math.fsum(prior['values']);assert saved_accuracy==summary['accuracy']
  out.update(complete=True,ordered_ray_parity=True,ordered_ray_sha256=digest.hexdigest(),stats_parity=True,no_tie_guard_branch=True,saved_value_diagnostic=dict(accuracy=saved_accuracy,bound=math.fsum(prior['bounds']),not_a_fresh_mass_estimate=True,known_mass_work_to_charge=summary['known_float_named_total'],prior_exact_setup_separate=True),additional_guard_numeric_units=sum(capture['guard_counts'].get(k,0) for k in ('scalar_equalities','boolean_reductions','integer_orientation_comparisons')),scope='Model-facing capture completed. Existing deterministic values replayed only after exact ray parity; earlier mass cost remains charged, not free advice. Inherited geometry proxies and cold/nonnumeric/native categories incomplete. No full-cost/noisy-rung/FP claim.')
 except Exception as exc:out.update(failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
 out.update(cpu_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (P/f'capture_seed{seed}.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
 if not out['complete']:raise RuntimeError(out['failure'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('seed',type=int,choices=(0,1));main(p.parse_args().seed)
