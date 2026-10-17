"""Counted independent DD geometry specialized to H2T6, no mass integration."""
from pathlib import Path
import numpy as np
from ..accounting import Audit,instrument_dd
from .bounded_rank import bounded_rank
DD=Path('/workspace/clean-neurons/m16/approaches/exact_ladder/cones.py')
OUTPUT_BASE=80;ORTHANT_BASE=100
GATE_IDS={0}|{1+16*(t-1)+k for t in range(1,6) for k in range(2)}
OUTPUT_IDS=set(range(80,86));RANK_IDS=set(range(86,92));ORTHANT_IDS=set(range(100,106))
assert len(GATE_IDS|OUTPUT_IDS|RANK_IDS|ORTHANT_IDS)==sum(map(len,(GATE_IDS,OUTPUT_IDS,RANK_IDS,ORTHANT_IDS)))
assert max(GATE_IDS|OUTPUT_IDS|RANK_IDS|ORTHANT_IDS)<128

def determinant(A,audit):
 a=np.array(A,dtype=float,copy=True);n=len(a);sgn=1.
 if n!=6:raise ValueError('T6 determinant only')
 audit.add('determinant_control',512)
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

def module(audit,validate_rank=False):
 dd=instrument_dd(DD,audit)
 def rank(M,ops):
  audit.add('rank_result_test',1)
  try:value=bounded_rank(M,audit)
  except ArithmeticError as exc:exc.rank_matrix=M.copy();raise
  audit.events['bounded_rank_calls']+=1
  if validate_rank:
   audit.validation_kernels['matrix_rank_SVD']+=1
   expected=np.linalg.matrix_rank(M,tol=1e-9)
   if value!=expected:raise AssertionError(f'bounded rank{value} != SVD{expected}')
  return value
 dd._rank=rank
 return dd

def outcome(dd,cell,model,ops,audit):
 R,Z,A,_=cell;Wo=model[2];T=6;L=Wo@A;audit.add('outcome_maps',2*T*A.shape[0]*T)
 dead=not A.any();audit.add('outcome_predicates',2*A.size+1);good=[]
 for j in range(T):
  audit.add('outcome_control',5)
  if dead:
   if j!=0:continue
   Q=(R,Z)
  else:
   Q=(R,Z)
   for k in range(T):
    audit.add('outcome_control',2)
    if k==j:continue
    margin=L[j]-L[k];audit.add('outcome_predicates',3*T)
    if not np.any(margin):
     if k<j:Q=None;break
     continue
    Q,_=dd.clip(Q[0],Q[1],margin,OUTPUT_BASE+k,ops,want=(True,False))
    if Q is None:break
   if Q is None:continue
  states=[(Q,0)]
  for k in range(T):
   if k==j:continue
   a=np.zeros(T);a[j]=1;a[k]=-1;next_states=[];audit.add('outcome_control',T+5)
   for P,c in states:
    ins,outs=dd.clip(P[0],P[1],a,OUTPUT_BASE+T+k,ops);audit.add('outcome_control',10)
    if ins is not None:next_states.append((ins,c))
    if outs is not None and c==0:next_states.append((outs,1))
   states=next_states
   if not states:break
  for P,c in states:
   audit.add('outcome_control',2)
   if c==1:good.append(P)
 return good

def simplices(dd,R,Z,ops,audit):
 pieces=[(R,Z)]
 for k in range(6):
  new=[]
  for r,z in pieces:
   sides=dd.clip(r,z,np.eye(6)[k],ORTHANT_BASE+k,ops);audit.add('orthant_control',40)
   for p in sides:
    if p is not None:
     if len(p[0])>=6:new.append(p)
     else:audit.events['lower_dimensional_orthant_sides']+=1
  pieces=new
 rays=[];det_sum=0.
 for r,z in pieces:
  audit.add('ray_l1_normalization',3*r.size);r=r/np.sum(abs(r),axis=1)[:,None]
  for idx in dd.triangulate(r,z,ops):
   ray=r[list(idx)].T;det=determinant(ray,audit);audit.add('simplex_test',1)
   if det!=0:rays.append(ray);det_sum+=abs(det);audit.add('determinant_sum',2)
   else:audit.events['zero_determinant_simplices']+=1
 return rays,det_sum
