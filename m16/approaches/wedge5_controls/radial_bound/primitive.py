"""Exact stored-ray ranges and outward bounds for ideal five-dimensional GL rules."""
from fractions import Fraction as F
from collections import Counter
from pathlib import Path
import importlib.util,sys,hashlib,json,time
P=Path(__file__).resolve().parent
class Exact:
 def __init__(self):self.counts=Counter();self.max_bits=0
 def record(self,x):
  self.max_bits=max(self.max_bits,abs(x.numerator).bit_length(),x.denominator.bit_length());return x
 def add(self,a,b):self.counts['add']+=1;return self.record(a+b)
 def sub(self,a,b):self.counts['subtract']+=1;return self.record(a-b)
 def mul(self,a,b):self.counts['multiply']+=1;return self.record(a*b)
 def div(self,a,b):self.counts['divide']+=1;return self.record(a/b)
 def total(self,values):
  s=F(0)
  for v in values:s=self.add(s,v)
  return s
 def solve(self,A,b):
  n=len(b);M=[list(row)+[rhs] for row,rhs in zip(A,b)]
  for j in range(n):
   pivot=next(i for i in range(j,n) if M[i][j]);M[j],M[pivot]=M[pivot],M[j];v=M[j][j];M[j]=[self.div(x,v) for x in M[j]]
   for i in range(n):
    if i!=j:
     v=M[i][j];M[i]=[self.sub(x,self.mul(v,y)) for x,y in zip(M[i],M[j])]
  return [row[-1] for row in M]
 def determinant(self,A):
  n=len(A);M=[list(row) for row in A];d=F(1)
  for j in range(n):
   pivot=next((i for i in range(j,n) if M[i][j]),None)
   if pivot is None:return F(0)
   if pivot!=j:M[j],M[pivot]=M[pivot],M[j];d=self.sub(F(0),d)
   v=M[j][j];d=self.mul(d,v)
   for i in range(j+1,n):
    c=self.div(M[i][j],v)
    for k in range(j+1,n):M[i][k]=self.sub(M[i][k],self.mul(c,M[j][k]))
  return d

def certificate(rays_hex):
 t=time.process_time();Q=Exact();A=[[Q.record(F(float.fromhex(x))) for x in row] for row in rays_hex];assert len(A)==5 and all(len(row)==5 for row in A)
 G=[[Q.total(Q.mul(x,y) for x,y in zip(a,b)) for b in A] for a in A];det=Q.determinant(A);assert det
 b=max(G[i][i] for i in range(5));acute=all(G[i][j]>=0 for i in range(5) for j in range(i))
 extra={}
 if acute:
  a=Q.div(min(G[i][i] for i in range(5)),F(5));mode='acute'
 else:
  candidates=[]
  for mask in range(1,32):
   S=[i for i in range(5) if mask>>i&1];z=Q.solve([[G[i][j] for j in S] for i in S],[F(1)]*len(S));den=Q.total(z);assert den>0
   y=[Q.div(x,den) for x in z]
   if min(y)>=0:candidates.append((Q.div(F(1),den),S,y))
  a,S,w=min(candidates,key=lambda x:x[0]);y=[F(0)]*5
  for i,v in zip(S,w):y[i]=v
  slack=[Q.sub(Q.total(Q.mul(G[i][j],y[j]) for j in range(5)),a) for i in range(5)]
  assert Q.total(y)==1 and min(y)>=0 and min(slack)>=0 and all(Q.mul(y[i],slack[i])==0 for i in range(5))
  extra=dict(y=list(map(str,y)),slack=list(map(str,slack)),feasible_faces=len(candidates),examined_faces=31);mode='exact_kkt'
 assert 0<a<=b
 return dict(a=str(a),b=str(b),det=str(det),gram=[[str(v) for v in row] for row in G],mode=mode,counts=dict(Q.counts),max_bits=Q.max_bits,cpu=time.process_time()-t,**extra)

def load_directed():
 p=json.loads((P/'protocol.json').read_text());path=Path(p['directed_source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==p['directed_sha256']
 name='wedge5_pinned_directed';spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class Bound:
 def __init__(self):
  self.module=load_directed();self.D=self.module.Directed(50);D=self.D;self.den=D.mul(D.point(32),D.power(D.pi(),2));self.cold_counts=dict(D.counts)
 def inv52(self,x):
  D=self.D;return D.div(D.point(1),D.mul(D.power(x,2),D.sqrt(x)))
 def one(self,cert,n):
  assert isinstance(n,int) and n>=2
  D=self.D;before=Counter(D.counts);a0=D.rational(F(cert['a']));b0=D.rational(F(cert['b']));a=D.point(a0.lo);b=D.point(b0.hi);assert 0<a.lo<=b.lo
  det=D.rational(abs(F(cert['det'])));N=n-2;coefficient_counts=Counter()
  if a.lo==b.lo:
   raw_tail=D.point(0);tail=raw_tail;r=D.point(0)
  else:
   c=D.div(D.add(a,b),D.point(2));r=D.div(D.sub(b,a),D.add(b,a));assert r.lo>=0 and r.hi<1
   partial=D.point(1);term=D.point(1);coef=F(1)
   for k in range(1,N+1):
    coef=coef*F(2*k+3,2*k);coefficient_counts['rational_multiply']+=1
    term=D.mul(term,r);partial=D.add(partial,D.mul(D.rational(coef),term))
   raw_tail=D.sub(self.inv52(a),D.mul(self.inv52(c),partial))
   assert raw_tail.hi>=0
   tail=self.module.Interval(max(raw_tail.lo,D.point(0).lo),raw_tail.hi)
  result=D.div(D.mul(det,tail),self.den)
  return dict(order=n,degree=N,a_expanded=str(a.lo),b_expanded=str(b.hi),r_interval=[str(r.lo),str(r.hi)],tail_interval=[str(tail.lo),str(tail.hi)],raw_tail_interval=[str(raw_tail.lo),str(raw_tail.hi)],lower_formula_endpoint=str(result.lo),upper_bound=str(result.hi),qualifies=F(result.hi)<=F(1,10**10),counts=dict(Counter(D.counts)-before),coefficient_counts=dict(coefficient_counts))
