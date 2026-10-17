"""Small fixed-width numerical rank with bounded auditable arithmetic.
Fail closed when numerical singular-value bounds do not resolve the threshold.
This is not interval arithmetic; the safety cushion is explicitly heuristic.
"""
import math
import numpy as np
TRANS=20
class AmbiguousRank(ArithmeticError):pass

def norm(x,audit):
 audit.add('qr_norm',2*x.size+TRANS)
 return math.sqrt(float(np.sum(x*x)))

def inverse(A,audit):
 n=len(A);v=np.concatenate([np.array(A,copy=True),np.eye(n)],axis=1)
 for j in range(n):
  pivot=j+int(np.argmax(np.abs(v[j:,j])));audit.add('qr_inverse',2*(n-j)+2)
  if v[pivot,j]==0:raise AmbiguousRank('singular selected coefficient minor')
  if pivot!=j:v[[j,pivot]]=v[[pivot,j]]
  v[j]/=v[j,j];audit.add('qr_inverse',2*n)
  for i in range(n):
   if i!=j:v[i]-=v[i,j]*v[j];audit.add('qr_inverse',4*n)
 return v[:,n:]

def bounded_rank(M,audit,tol=1e-9):
 A=np.asarray(M,dtype=float);m,n=A.shape;R=A.copy();Q=[];selected=[]
 if n>5:raise ValueError('this bounded control allowance is for width at most5')
 # At most6 outer iterations,5x5 inversion, and20 paired reorthogonalization
 # visits. Covers explicit scalar branch/loop/index arithmetic not included
 # in matrix/reduction counters; excludes interpreter and memory addressing.
 audit.add('qr_control_static',1024)
 size=norm(A,audit);cushion=1e-12*max(1.,size);audit.add('qr_bounds',3)
 for step in range(min(m,n)+1):
  norms=np.sum(R*R,axis=1);residual=math.sqrt(float(norms.sum()));audit.add('qr_pivot',2*m*n+m+TRANS+5)
  # Candidate proposal only: final retained/omitted brackets still decide rank.
  if residual<=tol or step==min(m,n):
   rank=len(Q)
   if rank==0:
    if size+cushion<tol:return 0
    raise AmbiguousRank('zero-rank bracket unresolved')
   Q=np.array(Q);C=A@Q.T;E=A-C@Q;G=Q@Q.T-np.eye(rank)
   audit.add('qr_bracket_products',4*m*n*rank+m*n+2*rank*rank*n+rank*rank)
   err=norm(E,audit);orth=norm(G,audit)
   X=inverse(C[selected],audit);Ires=np.eye(rank)-C[selected]@X
   audit.add('qr_bracket_products',2*rank**3+rank*rank)
   inv_err=norm(Ires,audit);inv_norm=norm(X,audit)
   if orth+cushion>=1 or inv_err+cushion>=1:raise AmbiguousRank('inverse/orthogonality bound unresolved')
   lower=math.sqrt(1-orth-cushion)*(1-inv_err-cushion)/inv_norm-err-cushion
   upper=err+cushion;audit.add('qr_bounds',TRANS+15)
   if lower>tol and (rank==min(m,n) or upper<tol):return rank
   raise AmbiguousRank(f'rank{rank} threshold unresolved: lower={lower}, upper={upper}, tol={tol}')
  pivot=int(np.argmax(norms));audit.add('qr_pivot',m)
  q=R[pivot].copy()
  # Explicit two-pass reorthogonalization of the new vector.
  for repeat in range(2):
   for old in Q:
    q-=float(q@old)*old;audit.add('qr_reorthogonalize',4*n)
  scale=norm(q,audit)
  if scale==0:raise AmbiguousRank('zero pivot before residual termination')
  q/=scale;audit.add('qr_normalize',n);Q.append(q);selected.append(pivot)
  for repeat in range(2):
   products=R@q;R-=products[:,None]*q;audit.add('qr_project',4*m*n)
 raise AssertionError('finite rank loop exhausted')
