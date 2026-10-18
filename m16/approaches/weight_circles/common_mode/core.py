"""Exact common-shift intervals; no model forward calls or probability inference."""
from collections import Counter
from fractions import Fraction as F

class Arithmetic:
 def __init__(self):self.counts=Counter();self.bits=0
 def keep(self,x):
  if isinstance(x,F):bits=max(abs(x.numerator).bit_length(),x.denominator.bit_length())
  else:bits=abs(x).bit_length()
  self.bits=max(self.bits,bits);return x
 def add(self,a,b):self.counts['add']+=1;return self.keep(a+b)
 def sub(self,a,b):self.counts['sub']+=1;return self.keep(a-b)
 def mul(self,a,b):self.counts['mul']+=1;return self.keep(a*b)
 def div(self,a,b):self.counts['rational_div']+=1;return self.keep(F(a)/F(b))
 def absolute(self,a):self.counts['abs']+=1;return self.keep(abs(a))
 def total(self,values):
  result=0
  for x in values:result=self.add(result,x)
  return result
 def dot(self,a,b):
  assert len(a)==len(b);return self.total(self.mul(x,y) for x,y in zip(a,b))
 def norm1(self,a):return self.total(self.absolute(x) for x in a)

def reconstruct(m,x,history,A):
 H,T,D=m['H'],m['T'],m['D'];wi,wh,wo=m['wi'],m['wh'],m['wo']
 assert len(x)==T and len(set(x))==T and len(history)==T and D>0
 assert len(wi)==len(wh)==H and len(wo)==T and all(len(r)==H for r in wh+wo)
 J=[[0]*T for _ in range(H)];scale=1;gate_rows=[];lo=hi=None
 for t,mask in enumerate(history):
  assert len(mask)==H and all(type(v) is bool for v in mask)
  B=[]
  for i in range(H):
   row=[]
   for j in range(T):
    v=A.dot(wh[i],[J[k][j] for k in range(H)])
    if j==t:v=A.add(v,A.mul(wi[i],scale))
    row.append(v)
   B.append(row);v=A.dot(row,x);s=A.total(row)
   if not any(row):assert v==s==0 and not mask[i];continue
   assert v!=0 and mask[i]==(v>0)
   gate_rows.append(dict(t=t,i=i,row=row,active=mask[i],intercept=v,slope=s))
   if s:
    root=A.div(A.sub(0,v),s)
    positive_slope=(s>0)==mask[i]
    if positive_slope:lo=root if lo is None else max(lo,root)
    else:hi=root if hi is None else min(hi,root)
  J=[r if mask[i] else [0]*T for i,r in enumerate(B)];scale=A.mul(scale,D)
 L=[[A.dot(row,[J[k][j] for k in range(H)]) for j in range(T)] for row in wo]
 den=A.mul(scale,D)
 assert (lo is None or lo<0) and (hi is None or hi>0)
 return dict(L=L,denominator=den,gate_rows=gate_rows,lower=lo,upper=hi)

def classify(x,L,gates,c,A):
 y=[A.add(v,c) for v in x];values=[A.dot(row,y) for row in L]
 winner=max(range(len(values)),key=lambda j:values[j]);target=sorted(range(len(x)),key=lambda j:x[j])[-2]
 tied=[j for j in range(len(L)) if j!=winner and values[j]==values[winner] and L[j]!=L[winner]]
 robust=not tied;radius=None
 if robust:
  candidates=[F(1)]
  for g in gates:
   z=A.dot(g['row'],y);assert z!=0 and (z>0)==g['active']
   candidates.append(A.div(A.absolute(z),A.mul(2,A.norm1(g['row']))))
  for i in range(len(x)):
   for j in range(i):candidates.append(A.div(A.absolute(A.sub(x[i],x[j])),4))
  for j in range(len(L)):
   if L[j]==L[winner]:continue
   gap=A.sub(values[winner],values[j]);assert gap>0
   row=[A.sub(a,b) for a,b in zip(L[winner],L[j])]
   candidates.append(A.div(gap,A.mul(2,A.norm1(row))))
  radius=min(candidates);assert radius>0
 return dict(c=c,y=y,values=values,prediction=winner,target=target,correct=winner==target,robust=robust,radius=radius,nonidentical_winner_ties=tied)

def analyse(x,r,A):
 L=r['L'];lo,hi=r['lower'],r['upper'];a=[A.dot(row,x) for row in L];b=[A.total(row) for row in L];roots=set()
 for i in range(len(L)):
  for j in range(i):
   slope=A.sub(b[i],b[j]);intercept=A.sub(a[i],a[j])
   if slope:
    c=A.div(A.sub(0,intercept),slope)
    if (lo is None or c>lo) and (hi is None or c<hi):roots.add(c)
 roots=sorted(roots);cuts=[lo]+roots+[hi];probes=[]
 for left,right in zip(cuts,cuts[1:]):
  if left is None:c=F(0) if right is None else A.sub(right,1)
  elif right is None:c=A.add(left,1)
  else:c=A.div(A.add(left,right),2)
  assert (left is None or c>left) and (right is None or c<right)
  q=classify(x,L,r['gate_rows'],c,A);q.update(kind='open_interval',lower=left,upper=right);probes.append(q)
 points=[classify(x,L,r['gate_rows'],c,A) for c in roots]
 pair=None
 for i,p in enumerate(probes):
  for j in range(i+1,len(probes)):
   q=probes[j]
   if p['robust'] and q['robust'] and p['correct']!=q['correct']:
    pair=[i,j];break
  if pair is not None:break
 return dict(lower=lo,upper=hi,logit_intercepts=a,logit_slopes=b,roots=roots,open_probes=probes,crossing_points=points,robust_opposite_pair=pair,line_open_change=len({p['correct'] for p in probes})>1,counts=dict(A.counts),max_integer_bits=A.bits)

def pack(x):
 if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
 if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [pack(v) for v in x]
 return x
