"""Primitive and original H2T5 geometry falsifiers; no model sweeps."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from ..accounting import Audit
from .geometry import determinant,enumerate_correct
from .volume import outer_rule,segment,volume
from .bounded_rank import bounded_rank,AmbiguousRank
HERE=Path(__file__).resolve().parent;ROOT=Path('/workspace/clean-neurons')
def powers(total,k):
 if k==1:yield(total,);return
 for x in range(total+1):
  for tail in powers(total-x,k-1):yield(x,)+tail
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--geometry',action='store_true');a=parser.parse_args()
 start=time.process_time();out={};audit=Audit();y,rest,w=outer_rule(4,audit);errors=[]
 for degree in range(5):
  for alpha in powers(degree,5):
   vals=np.prod(y**np.array(alpha[:3]),axis=1)*rest**(alpha[3]+alpha[4])
   vals*=math.factorial(alpha[3])*math.factorial(alpha[4])/math.factorial(alpha[3]+alpha[4]+1)
   expected=math.prod(math.factorial(v) for v in alpha)/math.factorial(4+degree)
   errors.append(abs(float(w@vals)-expected))
 assert max(errors)<1e-14
 out['degree_check']=dict(order=4,barycentric_max_degree=4,monomials=len(errors),max_absolute_error=max(errors),rule_cost=audit.report())
 ref=json.loads((ROOT/'m16/approaches/small_ladder_review/t5_independent/refined_results.json').read_text())
 segment_errors=[]
 for row in ref['segment_controls']:
  b=Audit();actual=float(segment(np.array(row['a']),np.array(row['b']),b));err=abs(actual-row['reference'])/abs(row['reference']);assert err<2e-14;segment_errors.append(err)
 out['segment_checks']=dict(cases=len(segment_errors),max_relative_error=max(segment_errors),reference='T52 independent100-digit antiderivative saved controls')
 rng=np.random.default_rng(52051);dets=[]
 for i in range(20):
  m=rng.normal(size=(5,5))
  if i%2:m[:,4]=m[:,0]+1e-8*m[:,4]
  b=Audit();err=abs(determinant(m,b)-np.linalg.det(m));dets.append(err);assert err<1e-11
 out['determinants']=dict(seed=52051,cases=20,max_absolute_error=max(dets))
 for last,rank in ((0.,4),(1e-6,5),(1e-9,None)):
  b=Audit();m=np.diag([1.,1.,1.,1.,last])
  try:
   r=bounded_rank(m,b);assert rank is not None and r==rank
  except AmbiguousRank:assert rank is None
 out['rank_fixtures']='rank4,rank5,and explicit threshold ambiguity passed'
 out['orthant']=[]
 for n in (8,16):
  b=Audit();p,bound=volume([np.eye(5)],n,b);assert abs(p-1/32)<bound+1e-13
  out['orthant'].append(dict(order=n,estimate=p,bound=bound,cost=b.report()))
 if a.geometry:
  import torch
  state=torch.load(ROOT/'work/checkpoints/H2_T5_seed0.pth',weights_only=True,map_location='cpu');model=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=(model[0].ravel(),*model[1:])
  b=Audit();begin=time.process_time();rays,stats,_=enumerate_correct(model,b,validate_rank=True)
  assert stats['gate_cells']==110 and len(rays)==2039
  out['base_geometry']=dict(stats=stats,correct_simplices=len(rays),cost=b.report(),cpu_including_rank_validation=time.process_time()-begin)
 out['cpu_seconds']=time.process_time()-start;(HERE/'checks.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(cpu=out['cpu_seconds'],degree=out['degree_check'],segment=out['segment_checks'],base=out.get('base_geometry'))))
if __name__=='__main__':main()
