"""Full counted curves on three previously specified H2T5 models, no new MC."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from ..accounting import Audit
from .geometry import enumerate_correct
from .volume import volume
HERE=Path(__file__).resolve().parent;ROOT=Path('/workspace/clean-neurons')
T48=Path('/workspace/work/recover-small_h2_56/m16/approaches/small_h2_56')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def model_hash(model):
 h=hashlib.sha256()
 for array in model:
  a=np.asarray(array,dtype='<f8');h.update(str(a.shape).encode('ascii'));h.update(a.tobytes(order='C'))
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['base','rms0.0025','rms0.02']);parser.add_argument('--output',default='results.json');args=parser.parse_args()
 start=time.process_time();checkpoint=ROOT/'work/checkpoints/H2_T5_seed0.pth'
 state=torch.load(checkpoint,weights_only=True,map_location='cpu');base=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
 refpath=ROOT/'m16/approaches/small_ladder_review/t5_independent/refined_results.json';refbase=json.loads(refpath.read_text());refnoisy=json.loads((T48/'cycle5.json').read_text())
 assert sha(checkpoint)==refbase['checkpoint_sha256']
 cases=[('base',base,refbase['cases'][0]['accuracy'])]
 for sigma in (.0025,.02):
  rng=np.random.default_rng(1)
  model=tuple(w+rng.normal(size=w.shape)*np.sqrt(np.mean(w*w))*sigma for w in base)
  reference=next(r for r in refnoisy['integrals'] if r['name']==f't5_sigma{sigma}_cones')
  cases.append((f'rms{sigma}',model,reference))
 if args.case:cases=[c for c in cases if c[0]==args.case]
 out=dict(provenance={str(p):sha(p) for p in (checkpoint,refpath,T48/'cycle5.json',T48/'models.py',HERE/'bounded_rank.py',HERE/'geometry.py',HERE/'volume.py')},random_model_inputs=0,models=[])
 def save():
  out['total_cpu_seconds']=time.process_time()-start;(HERE/args.output).write_text(json.dumps(out,indent=2)+'\n')
 for name,model,ref in cases:
  row=dict(name=name,model_sha256=model_hash(model),reference=ref['estimate'],reference_bound=ref['conditional_quadrature_error_bound'],curves=[]);out['models'].append(row);audit=Audit();active=audit;begin=time.process_time()
  try:
   rays,stats,_=enumerate_correct(model,audit);geometry_cpu=time.process_time()-begin
   row.update(geometry=stats,correct_simplices=len(rays),geometry_cpu=geometry_cpu,geometry_cost=audit.report());save()
   for n in (16,24,32):
    cost=copy.deepcopy(audit);active=cost;begin=time.process_time();p,bound=volume(rays,n,cost);cpu=time.process_time()-begin
    equivalent=cost.known/65;variance=ref['estimate']*(1-ref['estimate'])/equivalent
    c=dict(order=n,accuracy=p,integration_only_bound=bound,difference_from_reference=p-ref['estimate'],within_combined_bounds=abs(p-ref['estimate'])<=bound+row['reference_bound']+1e-11,cost=cost.report(),mass_cpu=cpu,end_to_end_cpu=geometry_cpu+cpu,affine_only_forward_scalar_ops=65,affine_only_forward_equivalents=equivalent,sampling_variance=variance,integration_bound_squared_over_sampling_variance=bound*bound/variance,quadrature_nodes=len(rays)*n**3)
    row['curves'].append(c);save();print(name,n,p,bound,cost.known,c['integration_bound_squared_over_sampling_variance'],flush=True)
    if time.process_time()-start>105:raise TimeoutError('bounded cycle stop')
  except Exception as exc:
   row.update(error_type=type(exc).__name__,error=str(exc),partial_cost=active.report(),cpu_to_failure=time.process_time()-begin);save();print(name,type(exc).__name__,str(exc),flush=True)
  if time.process_time()-start>110:break
 save();print('TOTAL_CPU',out['total_cpu_seconds'],flush=True)
if __name__=='__main__':main()
