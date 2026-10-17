"""Two existing checkpoints, full accounting curves; no new noise/model sweep."""
import copy,json,time
from pathlib import Path
import numpy as np
from accounting import Audit,enumerate_correct,volume
HERE=Path(__file__).resolve().parent;ROOT=Path('/workspace/clean-neurons')
DD=ROOT/'m16/approaches/exact_ladder/cones.py'

def forward(model,x):
 wi,wh,wo=model;h=np.zeros((len(x),len(wi)))
 for k in range(4):h=np.maximum(h@wh.T+x[:,k,None]*wi,0)
 return h@wo.T

def main():
 import torch
 start=time.process_time();report=dict(convention='Known scalar upper bound + explicit matrix_rank black boxes; TRANS=20',models=[])
 for H in (2,4):
  state=torch.load(ROOT/f'work/checkpoints/H{H}_T4_seed0.pth',map_location='cpu',weights_only=True)
  model=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=(model[0].ravel(),model[1],model[2])
  audit=Audit();begin=time.process_time();rays,stats=enumerate_correct(model,DD,audit);geometry_cpu=time.process_time()-begin
  row=dict(H=H,T=4,geometry=stats,correct_simplices=len(rays),geometry_cpu=geometry_cpu,geometry_cost=audit.report(),curves=[])
  # Timing only; this small batch is not used as an accuracy reference.
  begin=time.process_time();x=np.random.default_rng(52010+H).normal(size=(65536,4));forward(model,x).argmax(1);np.argsort(x,axis=1)[:,-2];sampling_cpu=time.process_time()-begin
  row['sampling_timing']=dict(n=len(x),cpu=sampling_cpu)
  for n in (4,8,12,16,20):
   a=copy.deepcopy(audit);begin=time.process_time();p,bound=volume(rays,n,a);mass_cpu=time.process_time()-begin
   fwd=2*(4*H*(H+1)+4*H);cost=a.known;svdcalls=sum(a.kernels.values());shape_price=sum(v*1000*k[1][0]*k[1][1]**2 for k,v in a.kernels.items())
   sensitivity=[]
   for label,extra in [('rank_price_zero_LOWER_BOUND_ONLY',0),('rank_price_1000_m_n_squared',shape_price),('rank_price_one_million_each',1000000*svdcalls),('rank_price_one_billion_each',1000000000*svdcalls)]:
    work=cost+extra;var=p*(1-p)/(work/fwd)
    sensitivity.append(dict(scenario=label,total_assumed_scalar_work=work,sampling_variance=var,integration_bound_mse_ratio=bound*bound/var))
   row['curves'].append(dict(order=n,accuracy=p,integration_only_bound=bound,cost=a.report(),mass_cpu=mass_cpu,end_to_end_cpu=geometry_cpu+mass_cpu,
     cpu_equivalent_samples=(geometry_cpu+mass_cpu)/sampling_cpu*len(x),conventional_forward_scalar_ops=fwd,rank_kernel_sensitivity=sensitivity,
     quadrature_nodes=len(rays)*n**3))
   print(H,n,p,cost,svdcalls,flush=True)
  row['reference_from_own_n20']=row['curves'][-1]['accuracy']
  for c in row['curves']:c['difference_from_n20']=c['accuracy']-row['reference_from_own_n20']
  report['models'].append(row);report['cpu_seconds']=time.process_time()-start
  (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 print('CPU',report['cpu_seconds'],flush=True)
if __name__=='__main__':main()
