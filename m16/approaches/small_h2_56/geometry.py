"""Bounded floating-point cone geometry. Ambiguous clipping raises explicitly."""
from itertools import combinations,product
import time
from math import gamma,pi
import numpy as np
from scipy.spatial import ConvexHull,QhullError

class Geometry:
 def __init__(self,seconds=30):
  self.start=time.process_time();self.seconds=seconds
  self.counts=dict(clips=0,hulls=0,vertex_plane_products=0,map_compositions=0,gate_pieces=0,correct_pieces=0,simplices=0,zero_determinants=0,bounded_degenerate_clips=0,omitted_mass_upper_bound=0.)
 def check(self):
  if time.process_time()-self.start>self.seconds:raise TimeoutError('geometry CPU cap')
 def hull(self,v):
  self.check();self.counts['hulls']+=1
  return ConvexHull(v[:,:-1])
 def clip(self,v,n):
  self.check();self.counts['clips']+=1
  norm=np.linalg.norm(n)
  if norm==0:return v
  n=n/norm;q=v@n;self.counts['vertex_plane_products']+=len(v)
  if q.min()>=-1e-12:return v
  if q.max()<=1e-12:return None
  hull=self.hull(v);edges=set()
  for face in hull.simplices:edges.update(tuple(sorted(e)) for e in combinations(face,2))
  keep=list(v[q>=-1e-12])
  for i,j in edges:
   if q[i]*q[j]<0:keep.append((q[i]*v[j]-q[j]*v[i])/(q[i]-q[j]))
  w=np.array(keep);_,idx=np.unique(np.round(w,12),axis=0,return_index=True);w=w[np.sort(idx)]
  if len(w)<v.shape[1]:return None
  try:h=self.hull(w)
  except QhullError as e:
   z=w[:,:-1]-w[:,:-1].mean(axis=0);_,_,rot=np.linalg.svd(z,full_matrices=False)
   widths=np.ptp(z@rot.T,axis=0);d=w.shape[1]
   # Add an outward numerical cushion, without claiming rigorous floating arithmetic.
   bound=gamma(d/2)*d**(d/2)/(2*pi**(d/2))*float(np.prod(widths+1e-14))
   if bound<=1e-12:
    self.counts['bounded_degenerate_clips']+=1;self.counts['omitted_mass_upper_bound']+=bound
    return None
   raise ArithmeticError(f'degenerate clip mass bound {bound}: refused discard') from e
  return w[h.vertices]
 def gates(self,model):
  wi,wh,wo=model;t=len(wo);eye=np.eye(t);cells=[]
  for signs in product((-1.,1.),repeat=t):
   pieces=[(eye*np.array(signs)[:,None],np.zeros((len(wi),t)))]
   for k in range(t):
    nxt=[]
    for v,a in pieces:
     z=wh@a+np.outer(wi,eye[k]);self.counts['map_compositions']+=1;split=[(v,[])]
     for row in z:
      sub=[]
      for p,g in split:
       if np.linalg.norm(row)==0:sub.append((p,g+[0]));continue
       pos=self.clip(p,row);neg=self.clip(p,-row)
       if pos is not None:sub.append((pos,g+[1]))
       if neg is not None:sub.append((neg,g+[0]))
      split=sub
     nxt.extend((p,np.array(g)[:,None]*z) for p,g in split)
    pieces=nxt
   cells.extend(pieces)
  self.counts['gate_pieces']=len(cells);return cells
 def correct(self,model,cells):
  wi,wh,wo=model;t=len(wo);eye=np.eye(t);out=[]
  for v,a in cells:
   logits=wo@a
   for c in range(t):
    p=v
    for j in range(t):
     if j==c:continue
     n=logits[c]-logits[j]
     if np.all(n==0) and j<c:p=None;break
     p=self.clip(p,n)
     if p is None:break
    if p is None:continue
    for m in range(t):
     if m==c:continue
     q=self.clip(p,eye[m]-eye[c])
     if q is None:continue
     for j in range(t):
      if j in(c,m):continue
      q=self.clip(q,eye[c]-eye[j])
      if q is None:break
     if q is not None:out.append(q)
  self.counts['correct_pieces']=len(out);return out
 def simplices(self,pieces):
  rays=[]
  for v in pieces:
   if len(v)==v.shape[1]:rays.append(v.T);continue
   h=self.hull(v);center=v.mean(axis=0)
   for face in h.simplices:
    ray=np.vstack([center,v[face]]).T
    if np.linalg.det(ray)==0:self.counts['zero_determinants']+=1
    else:rays.append(ray)
  self.counts['simplices']+=len(rays);return np.asarray(rays)
