"""Portable scalar-operation bounds with explicit opaque numerical kernels."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib,math,types
import numpy as np
TRANS=20

class Audit:
 def __init__(self):self.parts=Counter();self.kernels=Counter();self.events=Counter();self.validation_kernels=Counter()
 def add(self,part,n):self.parts[part]+=int(n)
 def kernel(self,name,shape):self.kernels[(name,tuple(shape))]+=1
 @property
 def known(self):return sum(self.parts.values())
 def report(self):return dict(known_arithmetic_upper_bound=self.known,parts=dict(self.parts),black_box_kernels=[dict(name=k[0],shape=list(k[1]),calls=v) for k,v in self.kernels.items()],events=dict(self.events),validation_kernel_calls=dict(self.validation_kernels))


def determinant(A,audit):
 """Scalar partial-pivot LU determinant; arithmetic, abs and comparisons charged."""
 a=np.array(A,dtype=float,copy=True);n=len(a);sgn=1.
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
 out=sgn
 for j in range(n):out*=a[j,j];audit.add('determinants',1)
 return float(out)


def gauss_rule(n,audit):
 """Gauss-Legendre on[0,1], scalar Newton setup with a finite iteration cap."""
 x=np.empty(n);w=np.empty(n)
 for i in range((n+1)//2):
  z=math.cos(math.pi*(i+.75)/(n+.5));audit.add('rule_newton',TRANS+4)
  for iteration in range(50):
   p0=1.;p1=z
   for k in range(2,n+1):
    p0,p1=p1,((2*k-1)*z*p1-(k-1)*p0)/k;audit.add('rule_newton',12)
   pp=n*(z*p1-p0)/(z*z-1);step=p1/pp;z-=step;audit.add('rule_newton',10)
   if abs(step)<3e-16:break
  else:raise ArithmeticError('Gaussian rule Newton did not converge')
  # Re-evaluate derivative at the final root, rather than stale Newton iterate.
  p0=1.;p1=z
  for k in range(2,n+1):p0,p1=p1,((2*k-1)*z*p1-(k-1)*p0)/k;audit.add('rule_newton',12)
  pp=n*(z*p1-p0)/(z*z-1);weight=1/((1-z*z)*pp*pp)
  x[i]=(1-z)/2;x[n-1-i]=(1+z)/2;w[i]=weight;w[n-1-i]=weight;audit.add('rule_newton',20)
 return x,w


def volume_rule(n,audit):
 x,w=gauss_rule(n,audit);u,v,z=np.meshgrid(x,x,x,indexing='ij');wu,wv,wz=np.meshgrid(w,w,w,indexing='ij')
 y=np.stack((u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z)),axis=-1).reshape(-1,4)
 weight=(wu*wv*wz*(1-u)**2*(1-v)).ravel();audit.add('rule_duffy',30*n**3)
 return y,weight


def volume(rays,n,audit):
 y,w=volume_rule(n,audit);total=0.;det_sum=0.
 for ray in rays:
  det=abs(determinant(ray,audit));audit.add('mass_scalar',1)
  q=np.sum((y@ray.T)**2,axis=1)
  total+=det/(2*np.pi**2)*np.dot(w,1/(q*q));det_sum+=det
  # 32 matrix/vector multiply-add scalar ops,7 norm,2 reciprocal-square,2 weighted sum.
  audit.add('mass_nodes',43*len(w));audit.add('mass_scalar',10)
 audit.add('bound',n+20)
 bound=det_sum/(6*np.pi**2)*(12*n+9)/3.**(n-1)
 return float(total),float(bound)


def instrument_dd(path,audit,rank_mode='svd',validate_rank=False):
 """Source is read once; geometry algorithms retained, local tie adapter below."""
 source=Path(path).read_text();dd=types.ModuleType('counted_dd');dd.__file__=str(path);exec(compile(source,str(path),'exec'),dd.__dict__)
 dd.source_sha256=hashlib.sha256(source.encode()).hexdigest();audit.add('module_bitset_setup',128*3)
 rawclip,rawlift,rawchildren,rawrank=dd.clip,dd.lift,dd._children,dd._rank
 def clip(R,Z,a,cid,ops,want=(True,True)):
  k,d=R.shape;NW=dd.NW;audit.events['clip_calls']+=1
  audit.add('zero_normal_check',2*d)
  if not np.any(a):return ((R,Z) if want[0] else None),None
  pairs=k*k//4
  # Worst branch sizes: <=floor(k²/4)opposite-sign pairs and <=k containment tests/pair.
  audit.add('clip_scalar',3*d+TRANS+2*k*d+20*k+50)
  audit.add('clip_adjacency',pairs*(6*NW+4)+pairs*k*(6*NW+6)+k*NW)
  audit.add('clip_new_rays',pairs*(8*d+TRANS+8*NW+20))
  return rawclip(R,Z,a,cid,ops,want)
 def lift(R,Z,a,alpha,cid,allmask,ops):
  k,d=R.shape;audit.events['lift_calls']+=1
  audit.add('lift_scalar',2*k*d+k+k*(3*(d+1)+TRANS)+4*k*dd.NW+20)
  return rawlift(R,Z,a,alpha,cid,allmask,ops)
 def children(R,Z,A,allm,t,model,ops):
  H=len(model[0]);audit.add('gate_map_setup',100*H*H+2*H*H*t+100*H*(t+1)+4*H*dd.NW)
  out=rawchildren(R,Z,A,allm,t,model,ops);audit.add('gate_masks',len(out)*(H*(t+1)+20));return out
 def rank(M,ops):
  audit.add('rank_result_test',1)
  if rank_mode=='svd':
   audit.kernel('matrix_rank_SVD',M.shape)
   return rawrank(M,ops)
  from bounded_rank import bounded_rank
  value=bounded_rank(M,audit);audit.events['bounded_rank_calls']+=1
  if validate_rank:
   expected=rawrank(M,ops);audit.validation_kernels['matrix_rank_SVD']+=1
   if value!=expected:raise AssertionError(f'QR rank{value} != SVD{expected} on{M.shape}')
  return value
 def triangulate(R,Z,ops):
  k,d=R.shape;audit.events['triangulate_calls']+=1
  inc=np.unpackbits(Z.view(np.uint8),axis=1,bitorder='little').astype(bool)
  used=np.nonzero(inc.any(0))[0];inc=inc[:,used];audit.add('tri_incidence',3*k*64*dd.NW+128)
  out=[]
  def rec(F,f):
   audit.add('tri_control',10)
   if len(F)==f:out.append(F);return
   v=F[0];sub=inc[list(F)];seen=set()
   for c in range(sub.shape[1]):
    audit.add('tri_face_scan',10*len(F)+50)
    col=sub[:,c]
    if col[0] or col.all() or not col.any():continue
    G=tuple(F[i] for i in np.nonzero(col)[0])
    if len(G)<f-1 or G in seen:continue
    seen.add(G)
    if dd._rank(R[list(G)],ops)!=f-1:continue
    n0=len(out);rec(G,f-1)
    for i in range(n0,len(out)):out[i]=(v,)+out[i];audit.add('tri_control',d+5)
  rec(tuple(range(k)),d);return out
 dd.clip=clip;dd.lift=lift;dd._children=children;dd._rank=rank;dd.triangulate=triangulate
 return dd


def outcome(dd,cell,model,ops,audit):
 """DRI outcome splitting with explicit first-index ownership for zero margins."""
 R,Z,A,_=cell;Wo=model[2];T=len(Wo);L=Wo@A;audit.add('outcome_maps',2*T*A.shape[0]*T);ops.add(2*T*A.shape[0]*T,'final')
 dead=not A.any();audit.add('outcome_predicates',2*A.size+1);good=[];bad=[]
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
    Q,_=dd.clip(Q[0],Q[1],margin,60+k,ops,want=(True,False));ops.add(T,'final')
    if Q is None:break
   if Q is None:continue
  S=[(Q,0)]
  for k in range(T):
   if k==j:continue
   a=np.zeros(T);a[j]=1;a[k]=-1;nS=[];audit.add('outcome_control',T+5)
   for P,c in S:
    ins,outs=dd.clip(P[0],P[1],a,60+T+k,ops);audit.add('outcome_control',10)
    if ins is not None:nS.append((ins,c))
    if outs is not None:
     if c+1>=2:bad.append(outs)
     else:nS.append((outs,c+1))
   S=nS
   if not S:break
  for P,c in S:(good if c==1 else bad).append(P);audit.add('outcome_control',2)
 return good,bad


def simplices(dd,R,Z,ops,audit):
 pieces=[(R,Z)]
 for k in range(4):
  new=[]
  for r,z in pieces:
   sides=dd.clip(r,z,np.eye(4)[k],90+k,ops);audit.add('orthant_control',20)
   new.extend(p for p in sides if p is not None and len(p[0])>=4)
  pieces=new
 out=[]
 for r,z in pieces:
  audit.add('ray_l1_normalization',3*r.size);r=r/np.sum(abs(r),axis=1)[:,None]
  for idx in dd.triangulate(r,z,ops):
   ray=r[list(idx)].T;det=determinant(ray,audit);audit.add('simplex_test',1)
   if det!=0:out.append(ray)
 return out


def enumerate_correct(model,dd_path,audit,rank_mode='svd',validate_rank=False):
 dd=instrument_dd(dd_path,audit,rank_mode,validate_rank);ops=dd.Ops();counts=[0]*4;rays=[];cells=0
 H=len(model[0]);assert H in (2,4) and len(model[2])==4
 if np.all(model[0]==0):raise ValueError('zero-input weights require separately proved special case')
 audit.add('root_gate_setup',100*H+100)
 for cell in dd.iter_cells(model,ops,counts):
  cells+=1;good,_=outcome(dd,cell,model,ops,audit)
  for R,Z in good:rays.extend(simplices(dd,R,Z,ops,audit))
 return rays,dict(gate_cells=cells,stage_counts=counts,source_sha256=dd.source_sha256,original_dd_counter=ops.n,original_dd_parts=ops.parts)
