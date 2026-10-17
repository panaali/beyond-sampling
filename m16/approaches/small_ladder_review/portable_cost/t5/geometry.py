"""T5 specialization of the already audited T4 DD wrappers."""
from pathlib import Path
import numpy as np
from ..accounting import Audit,instrument_dd,outcome
from .bounded_rank import bounded_rank
DD=Path('/workspace/clean-neurons/m16/approaches/exact_ladder/cones.py')

def determinant(A,audit):
 a=np.array(A,dtype=float,copy=True);n=len(a);sgn=1.
 if n!=5:raise ValueError('T5 determinant only')
 audit.add('determinant_control',256)
 for j in range(n):
  pivot=j;best=abs(a[j,j]);audit.add('determinants',1)
  for i in range(j+1,n):
   v=abs(a[i,j]);audit.add('determinants',2)
   if v>best:best=v;pivot=i
  audit.add('determinants',1)
  if best==0:return 0.
  if pivot!=j:a[[j,pivot]]=a[[pivot,j]];sgn=-sgn;audit.add('determinants',1)
  for i in range(j+1,n):
   f=a[i,j]/a[j,j];audit.add('determinants',1)
   for k in range(j+1,n):a[i,k]-=f*a[j,k];audit.add('determinants',2)
 for j in range(n):sgn*=a[j,j];audit.add('determinants',1)
 return float(sgn)

def triangulated(dd,R,Z,ops,audit):
 pieces=[(R,Z)]
 for k in range(5):
  new=[]
  for r,z in pieces:
   sides=dd.clip(r,z,np.eye(5)[k],90+k,ops);audit.add('orthant_control',30)
   new.extend(p for p in sides if p is not None and len(p[0])>=5)
  pieces=new
 out=[]
 for r,z in pieces:
  audit.add('ray_l1_normalization',3*r.size);r=r/np.sum(abs(r),axis=1)[:,None]
  for idx in dd.triangulate(r,z,ops):
   ray=r[list(idx)].T;det=determinant(ray,audit);audit.add('simplex_test',1)
   if det!=0:out.append(ray)
 return out

def enumerate_correct(model,audit,validate_rank=False,partition=False):
 dd=instrument_dd(DD,audit)
 def rank(M,ops):
  audit.add('rank_result_test',1)
  try:value=bounded_rank(M,audit)
  except ArithmeticError as exc:
   exc.rank_matrix=M.copy()
   raise
  audit.events['bounded_rank_calls']+=1
  if validate_rank:
   audit.validation_kernels['matrix_rank_SVD']+=1
   expected=np.linalg.matrix_rank(M,tol=1e-9)
   if value!=expected:raise AssertionError(f'bounded rank{value} != SVD{expected}')
  return value
 dd._rank=rank;ops=dd.Ops();counts=[0]*5;rays=[];gates=[];cells=0
 audit.add('root_gate_setup',400)
 if len(model[0])!=2 or model[1].shape!=(2,2) or model[2].shape!=(5,2):raise ValueError('H2T5 only')
 if np.all(model[0]==0):raise ValueError('zero input weights unsupported')
 for cell in dd.iter_cells(model,ops,counts):
  cells+=1;good,_=outcome(dd,cell,model,ops,audit)
  for R,Z in good:rays.extend(triangulated(dd,R,Z,ops,audit))
  if partition:gates.extend(triangulated(dd,cell[0],cell[1],ops,audit))
 return rays,dict(gate_cells=cells,stage_counts=counts,source_sha256=dd.source_sha256,original_dd_counter=ops.n,original_dd_parts=ops.parts),gates
