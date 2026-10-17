"""Independent counted H2T6 geometry, with separately charged partition audit."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from ..accounting import Audit
from .geometry import module,outcome,simplices
HERE=Path(__file__).resolve().parent

def main():
 start=time.process_time();path=Path('/workspace/clean-neurons/work/checkpoints/H2_T6_seed0.pth')
 state=torch.load(path,weights_only=True,map_location='cpu');model=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=(model[0].ravel(),*model[1:])
 prod=Audit();valid=Audit();begin=time.process_time();dd=module(prod);ops=dd.Ops()
 vbegin=time.process_time();check=module(valid);checkops=check.Ops();validation_cpu=time.process_time()-vbegin;io_cpu=0.
 prod.add('root_gate_setup',500);counts=[0]*6;rays=[];gate_simplices=0;gate_det=0.;correct_det=0.;max_affine=0.;processed=0;phase='setup';complete=False
 report=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=dd.source_sha256,random_model_inputs=0)
 def save():
  nonlocal io_cpu
  report.update(complete=complete,processed_gate_cells=processed,stage_counts=counts,correct_simplices=len(rays),gate_simplices=gate_simplices,gate_determinant_sum=gate_det,correct_determinant_sum=correct_det,max_affine_error=max_affine,production_cost=prod.report(),partition_validation_cost=valid.report(),production_cpu_excluding_validation_and_checkpoints=time.process_time()-begin-validation_cpu-io_cpu,partition_validation_cpu=validation_cpu,checkpoint_io_cpu=io_cpu,total_cpu_seconds=time.process_time()-start)
  before=time.process_time();(HERE/'geometry_results.json').write_text(json.dumps(report,indent=2)+'\n');io_cpu+=time.process_time()-before
 try:
  for cell in dd.iter_cells(model,ops,counts):
   phase='correct_regions';pieces=outcome(dd,cell,model,ops,prod)
   for R,Z in pieces:
    parts,det=simplices(dd,R,Z,ops,prod);rays.extend(parts);correct_det+=det
   phase='partition_validation';vbegin=time.process_time()
   try:
    parts,det=simplices(check,cell[0],cell[1],checkops,valid);gate_simplices+=len(parts);gate_det+=det
    x=cell[0].sum(axis=0);x/=np.sqrt(x@x);h=np.zeros(2)
    for z in x:h=np.maximum(model[1]@h+model[0]*z,0)
    max_affine=max(max_affine,float(np.max(abs(h-cell[2]@x))))
    valid.add('affine_validation',2*cell[0].size+300)
   finally:validation_cpu+=time.process_time()-vbegin
   processed+=1
   if processed%25==0:save();print('cells',processed,'correct_simplices',len(rays),'cpu',time.process_time()-start,flush=True)
   if time.process_time()-start>103:raise TimeoutError('bounded geometry cycle deadline')
  complete=True
  assert abs(gate_det-64)<1e-8,gate_det
  assert 0<=correct_det<=64+1e-8,correct_det
 except Exception as exc:
  report.update(error_type=type(exc).__name__,error=str(exc),failure_phase=phase)
  if hasattr(exc,'rank_matrix'):report['failing_rank_matrix']=exc.rank_matrix.tolist()
  print('FAIL',phase,type(exc).__name__,str(exc),flush=True)
 save()
 if rays:
  before=time.process_time();np.savez_compressed(HERE/'independent_correct_rays.npz',rays=np.array(rays));io_cpu+=time.process_time()-before
  report['ray_cache_sha256']=hashlib.sha256((HERE/'independent_correct_rays.npz').read_bytes()).hexdigest();save()
 print(json.dumps({k:report.get(k) for k in ('complete','processed_gate_cells','correct_simplices','gate_simplices','gate_determinant_sum','max_affine_error','total_cpu_seconds','error')}),flush=True)
if __name__=='__main__':main()
