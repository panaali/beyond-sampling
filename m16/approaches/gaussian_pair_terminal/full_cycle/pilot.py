import hashlib,json,time
from pathlib import Path
import torch
from driver import estimate
HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=a[0].ravel(),a[1],a[2]
reference=json.loads((HERE.parent.parent/'rank_conditioned_transport/direct_first_cycle/label_reference.json').read_text())
truth={str(r['label']):r['probability'] for r in reference['rows']}
start=time.process_time();out=dict(reference=.30773074302392744,rows=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE/'driver.py',HERE.parent/'terminal.py')})
for order in (2,4):
 for paired in (False,True):
  row=dict(order=order,paired=paired)
  try:
   value,stats=estimate(model,order,paired,seconds=100-(time.process_time()-start))
   errors=[stats['contributions'].get(j,0.)-truth[str(j)] for j in range(4)]
   row.update(estimate=value,error=value-out['reference'],per_label_errors=errors,component_l2=sum(e*e for e in errors)**.5,stats=stats)
  except Exception as exc:row['failure']=repr(exc)
  out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
  (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
  print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
  if 'failure' in row:raise SystemExit(1)
for paired in (False,True):
 a,b=[r for r in out['rows'] if r['paired']==paired]
 out['paired' if paired else 'baseline']=dict(refinement_change=abs(a['estimate']-b['estimate']),error=b['error'],cpu=b['stats']['cpu_seconds'],error_squared_cpu=b['error']**2*b['stats']['cpu_seconds'],component_l2=b['component_l2'])
(HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('baseline','paired','total_cpu_seconds')}))
