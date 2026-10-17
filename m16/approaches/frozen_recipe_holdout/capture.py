"""Fixed seed1 exact/recipe error geometry; no Gaussian integration."""
import time
START=time.process_time()
import hashlib,json,resource,sys
from pathlib import Path
import numpy as np
import torch
TASK=Path(__file__).resolve().parent
MODE=sys.argv[1]
assert MODE in ('exact','recipe108')
HERE=TASK/MODE
HERE.mkdir(exist_ok=True)
SOURCE=TASK.parent/'leaf_mass_review'
sys.path.insert(0,str(SOURCE))
M43=Path('/workspace/work/admin-t121-rank-mass/m16/approaches/arc_m43_repro/m43.py');sys.path.insert(0,str(M43.parent))
import estT_snapshot as E
assert Path(E.__file__).resolve()==SOURCE/'estT_snapshot.py'
WEIGHTS=Path('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_1.pth')
class Budget(Exception):pass
class Ops:
 def __init__(self):self.n=0
 def add(self,n):
  self.n+=n
  if self.n>200_000_000 or time.process_time()-START>55:raise Budget('55CPU/200M geometry cap')
def main():
 torch.set_num_threads(1);state=torch.load(WEIGHTS,map_location='cpu',weights_only=True);model=(state['rnn.weight_ih_l0'].numpy().astype(float)[:,0],state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float));ops=Ops();target=[]
 def record(rays,o):o.add(len(rays)**2);target.append(np.array(rays,copy=True));E.STATS['simplices']+=1;return 0.
 E.orthant=record;original=E._cell_error
 # All-off tie handling is an explicit first-index rule in pinned callback.
 base=E.Cone.from_normals(np.eye(4));g=np.zeros(8);A=np.zeros((8,4));original(base,g,A,model[2],0,ops);assert not target
 original(base,g,A,model[2],1,ops);assert len(target)==1;target=[]
 controls=dict(alloff_first_index_correct=True,alloff_other_index_wrong=True)
 all_rays=[];parent_ids=[];leaf_ids=[];counts=dict(leaves=0,alloff=0,active_identical_readout_pairs=0,earlier_identical_truth_pairs=0);perleaf=[]
 def terminal(c,g,A,Wo,truth,o):
  leaf=counts['leaves'];counts['leaves']+=1
  if not np.any(g):counts['alloff']+=1
  else:
   L=Wo@(g[:,None]*A)
   for a in range(4):
    for b in range(a):
     if np.array_equal(L[a],L[b]):
      counts['active_identical_readout_pairs']+=1
      if a==truth:counts['earlier_identical_truth_pairs']+=1
  begin=len(target);result=original(c,g,A,Wo,truth,o);perleaf.append((leaf,begin,len(target)));return result
 E._cell_error=terminal;E.MODE[0]=MODE;E.STATS.update({k:0 for k in E.STATS});halves=E.halves(4)
 out=dict(mode=MODE,checkpoint_seed=1,complete=False,controls=controls,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'estT_snapshot.py',TASK/'capture.py',M43,WEIGHTS)},parents=[],counts=counts,random_network_inputs=0)
 def save():
  out.update(cpu_seconds_including_imports=time.process_time()-START,named_geometry_ops=ops.n,stats=dict(E.STATS),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,completed_error_simplices=len(all_rays));np.savez_compressed(HERE/'error_captured.npz',rays=np.asarray(all_rays),parent_ids=np.asarray(parent_ids),leaf_ids=np.asarray(leaf_ids));(HERE/'error_capture.json').write_text(json.dumps(out,indent=2)+'\n')
 try:
  for index,h in enumerate(halves):
   target=[];perleaf=[];before=counts['leaves'];E._finish(h['cone'],(h['sigma']*model[0]>0).astype(float),np.outer(model[0],np.eye(4)[0]),1,model,h['truth'],h['n_true'],.1,ops,MODE=='exact')
   begin=len(all_rays);all_rays.extend(target);parent_ids.extend([index]*len(target))
   for leaf,a,b in perleaf:leaf_ids.extend([leaf]*(b-a))
   out['parents'].append(dict(index=index,permutation=h['perm'],sigma=h['sigma'],truth=h['truth'],leaves=counts['leaves']-before,begin=begin,end=len(all_rays)));save()
  out['complete']=True
 except Exception as err:
  out['failure']=repr(err);out['partial_current_parent_simplices_not_integrated']=len(target);save()
  if not isinstance(err,Budget):raise
 save();print(json.dumps({k:v for k,v in out.items() if k not in ('source_hashes','parents')}))
if __name__=='__main__':main()
