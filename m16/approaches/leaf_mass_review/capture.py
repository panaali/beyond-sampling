"""Fixed H8T4 gate-leaf capture with Gaussian quadrature disabled."""
import time
START=time.process_time()
import hashlib,importlib.util,json,math,resource,sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
M43=Path('/workspace/work/admin-t121-rank-mass/m16/approaches/arc_m43_repro/m43.py')
sys.path.insert(0,str(M43.parent))
import estT_snapshot as E
WEIGHTS=Path('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth')
class Budget(Exception):pass
class Ops:
 def __init__(self):self.n=0
 def add(self,k):
  self.n+=k
  if self.n>200_000_000 or time.process_time()-START>55:raise Budget('55CPU/200M named-geometry cap')
def detF(matrix):
 a=[list(map(F,row)) for row in matrix];answer=F(1)
 for j in range(len(a)):
  pivot=next((k for k in range(j,len(a)) if a[k][j]),None)
  if pivot is None:return F(0)
  if pivot!=j:a[j],a[pivot]=a[pivot],a[j];answer=-answer
  v=a[j][j];answer*=v
  for k in range(j+1,len(a)):
   c=a[k][j]/v
   for q in range(j+1,len(a)):a[k][q]-=c*a[j][q]
 return answer

def section(rays,anchor,exact=False):
 if exact:
  a=list(map(F,anchor));r=[list(map(F,row)) for row in rays];return abs(detF(r))/math.prod(sum(x*y for x,y in zip(row,a)) for row in r)
 d=rays@anchor;assert d.min()>0;return abs(float(np.linalg.det(rays)))/float(np.prod(d))

def main():
 torch.set_num_threads(1);state=torch.load(WEIGHTS,map_location='cpu',weights_only=True);model=(state['rnn.weight_ih_l0'].numpy().astype(float)[:,0],state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float));assert model[1].shape==(8,8) and model[2].shape==(4,8)
 ops=Ops();target=[];leaves=[]
 def record(rays,o):
  o.add(len(rays)**2);target.append(np.array(rays,copy=True));E.STATS['simplices']+=1;return 0.
 E.orthant=record
 def terminal(cone,g,A,Wo,truth,o):
  E.STATS['cells']+=1;begin=len(target);E.mass(cone,o);leaves.append(dict(begin=begin,end=len(target),rays=len(cone.R),facets=len(cone.N),minimum_own_inequality=float(np.min(cone.R@cone.N.T/np.linalg.norm(cone.N,axis=1)[None,:]))));return 0.
 E._cell_error=terminal
 # Analytic partition, with identical topology and independent exact section formula.
 eye=np.eye(4);base=E.Cone.from_normals(eye);inside,outside=E.clip(base,eye[0]-eye[1],ops);anchor=np.ones(4)
 target=[];E.mass(base,ops);parent=sum(section(r,anchor,True) for r in target);target=[]
 for c in (inside,outside):E.mass(c,ops)
 child=sum(section(r,anchor,True) for r in target);assert parent==child==1
 controls=dict(orthant_split_exact=True,normalized_parent_determinant=str(parent),child_simplices=len(target))
 target=[];leaves=[];E.STATS.update({k:0 for k in E.STATS});E.MODE[0]='exact';halves=E.halves(4);out=dict(complete=False,geometry_complete=False,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'estT_snapshot.py',HERE/'capture.py',M43,WEIGHTS)},controls=controls,parents=[],leaf_count=0,exact_checks=[],random_network_inputs=0);all_rays=[];all_parent_ids=[];all_leaf_ids=[];all_parents=[]
 def save():
  out.update(cpu_seconds_including_imports=time.process_time()-START,named_geometry_ops=ops.n,stats=dict(E.STATS),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  (HERE/'capture.json').write_text(json.dumps(out,indent=2)+'\n')
  np.savez_compressed(HERE/'captured.npz',rays=np.asarray(all_rays),parent_ids=np.asarray(all_parent_ids),leaf_ids=np.asarray(all_leaf_ids))
 try:
  for index,h in enumerate(halves):
   anchor=h['cone'].N.sum(axis=0);assert np.min(h['cone'].R@anchor)>0
   target=[];E.mass(h['cone'],ops);parents=target;all_parents.append((anchor,parents));parent_volume=sum(section(r,anchor) for r in parents)
   target=[];leaves=[];g=(h['sigma']*model[0]>0).astype(float);E._finish(h['cone'],g,np.outer(model[0],np.eye(4)[0]),1,model,h['truth'],h['n_true'],.1,ops,True)
   child_volume=math.fsum(section(r,anchor) for r in target);minimum_dot=min(float(np.min(r@anchor)) for r in target);parent_violation=min(float(np.min(r@h['cone'].N.T/np.linalg.norm(h['cone'].N,axis=1)[None,:])) for r in target)
   first=len(all_rays);all_rays.extend(target);all_parent_ids.extend([index]*len(target))
   for j,leaf in enumerate(leaves):all_leaf_ids.extend([out['leaf_count']+j]*(leaf['end']-leaf['begin']))
   row=dict(index=index,sigma=h['sigma'],perm=h['perm'],anchor=anchor.tolist(),parent_simplices=len(parents),leaf_count=len(leaves),child_simplices=len(target),parent_volume=parent_volume,child_volume=child_volume,relative_section_residual=child_volume/parent_volume-1,minimum_anchor_dot=minimum_dot,minimum_parent_inequality=parent_violation,minimum_leaf_inequality=min(r['minimum_own_inequality'] for r in leaves),begin=first,end=len(all_rays))
   out['leaf_count']+=len(leaves);out['parents'].append(row);save()
  out['geometry_complete']=True
  worst=max(range(len(out['parents'])),key=lambda i:abs(out['parents'][i]['relative_section_residual']))
  for i in dict.fromkeys((0,1,worst)):
   row=out['parents'][i];anchor,parents=all_parents[i];pv=sum(section(r,anchor,True) for r in parents);cv=F(0)
   for r in all_rays[row['begin']:row['end']]:
    ops.add(0);cv+=section(r,anchor,True)
   delta=cv/pv-1;out['exact_checks'].append(dict(parent=i,exact_zero=delta==0,relative_residual=float(delta),residual_numerator_bits=delta.numerator.bit_length(),residual_denominator_bits=delta.denominator.bit_length()))
  out['complete']=True
 except Exception as err:
  out['failure']=repr(err);save()
  if not isinstance(err,Budget):raise
 save();print(json.dumps({k:out[k] for k in ('complete','geometry_complete','leaf_count','exact_checks','cpu_seconds_including_imports','named_geometry_ops','stats')}));print('max section residual',max(abs(r['relative_section_residual']) for r in out['parents']))
if __name__=='__main__':main()
