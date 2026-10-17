"""Bounded-rank falsifiers and existing-checkpoint geometry equivalence only."""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
from accounting import Audit, enumerate_correct, volume
from bounded_rank import bounded_rank, AmbiguousRank
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons')

def fixtures():
 rng=np.random.default_rng(52031); rows=[]
 matrices=[('zero',np.zeros((7,4)),0),('identity',np.eye(4),4),('rank2',np.array([[1.,0,0,0],[0,1,0,0],[1,1,0,0]]),2)]
 for m,n,r in [(3,4,1),(9,4,2),(20,4,3),(7,4,4)]:
  matrices.append((f'factor_{m}_{n}_{r}',rng.normal(size=(m,r))@rng.normal(size=(r,n)),r))
 for label,s in [('below',1e-12),('above',1e-6),('threshold',1e-9)]:
  matrices.append((label,np.diag([1.,s]),None if label=='threshold' else (1 if label=='below' else 2)))
 for label,M,expected in matrices:
  a=Audit()
  try:
   rank=bounded_rank(M,a);out=dict(rank=rank)
   assert expected is not None and rank==expected,(label,rank,expected)
   assert rank==np.linalg.matrix_rank(M,tol=1e-9)
  except AmbiguousRank as e:
   assert expected is None,(label,str(e))
   out=dict(ambiguity=str(e))
  rows.append(dict(name=label,matrix=M.tolist(),expected=expected,cost=a.report(),**out))
 return rows

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--H',type=int,choices=[2,4]);args=parser.parse_args()
 start=time.process_time();out=dict(seed=52031,fixtures=fixtures())
 if args.H:
  import torch
  H=args.H;state=torch.load(ROOT/f'work/checkpoints/H{H}_T4_seed0.pth',weights_only=True,map_location='cpu')
  model=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=(model[0].ravel(),model[1],model[2])
  audit=Audit();begin=time.process_time()
  rays,stats=enumerate_correct(model,ROOT/'m16/approaches/exact_ladder/cones.py',audit,rank_mode='bounded_qr',validate_rank=True)
  geometry_cpu=time.process_time()-begin;geometry_audit=copy.deepcopy(audit)
  begin=time.process_time();p,b=volume(rays,16,audit);volume_cpu=time.process_time()-begin
  old=next(row for row in json.loads((HERE/'results.json').read_text())['models'] if row['H']==H)
  old16=next(c for c in old['curves'] if c['order']==16)
  assert len(rays)==old['correct_simplices']
  assert abs(p-old16['accuracy'])<1e-14,(p,old16['accuracy'])
  assert not audit.kernels
  out['model']=dict(H=H,geometry=stats,simplices=len(rays),geometry_cpu_including_svd_validation=geometry_cpu,volume_cpu=volume_cpu,accuracy=p,old_accuracy=old16['accuracy'],integration_only_bound=b,geometry_cost=geometry_audit.report(),total_cost=audit.report())
 out['cpu_seconds']=time.process_time()-start
 name=f'rank_checks_H{args.H}.json' if args.H else 'rank_fixtures.json'
 (HERE/name).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(file=name,cpu=out['cpu_seconds'],model=out.get('model'))))
if __name__=='__main__':main()
