"""Exact rational fixed-denominator numerator-angle error bounds."""
from collections import Counter
from fractions import Fraction as F

def serial(x):return None if x is None else [str(x.numerator),str(x.denominator)]
class Ledger:
 def __init__(self,values):
  self.count=Counter();self.peak=[0,0]
  for x in values:
   if isinstance(x,F):self.observe(x)
 def observe(self,x):
  self.peak[0]=max(self.peak[0],abs(x.numerator).bit_length());self.peak[1]=max(self.peak[1],x.denominator.bit_length());return x
 def op(self,name,a,b):
  self.count['rational_'+name]+=1
  return self.observe({'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[name]())
 def cmp(self,name,a,b):
  self.count['comparison_'+name]+=1
  return {'lt':lambda:a<b,'gt':lambda:a>b,'eq':lambda:a==b}[name]()
 def ab(self,x):self.count['rational_abs']+=1;return self.observe(abs(x))
 def result(self,status,**kw):return dict(status=status,counts=dict(self.count),peak_bits=self.peak,**kw)

def enclose(x,lo,hi,p):
 L=Ledger([x,lo,hi,p]);L.count['type_checks']=4
 if not all(isinstance(v,F) for v in [x,lo,hi,p]):return L.result('unknown_nonrational',bound=None)
 if L.cmp('gt',lo,hi):return L.result('invalid_interval',bound=None)
 if L.cmp('lt',p,lo) or L.cmp('gt',p,hi):return L.result('invalid_point_outside',bound=None)
 if L.cmp('lt',x,F(0)):return L.result('unsupported_negative_x',bound=None)
 positive=L.cmp('gt',lo,F(0));negative=L.cmp('lt',hi,F(0))
 if L.cmp('eq',x,F(0)):
  if positive or negative:return L.result('bounded_constant',bound=F(0),raw=F(0),uniform=F(0))
  return L.result('unknown_origin_possible',bound=None)
 a=L.op('sub',p,lo);b=L.op('sub',hi,p);radius=a if L.cmp('gt',a,b) else b
 d=L.ab(lo) if positive else (L.ab(hi) if negative else F(0))
 den=L.op('add',L.op('mul',x,x),L.op('mul',d,d));slope=L.op('div',x,den);raw=L.op('mul',slope,radius)
 bound=F(4) if L.cmp('gt',raw,F(4)) else raw
 # Separate analytic comparator, never selected in place of the frozen bound.
 uniform=None
 if L.cmp('gt',d,F(0)):
  uniform=L.op('div',radius,L.op('mul',F(2),d));assert not L.cmp('gt',raw,uniform)
 return L.result('bounded',bound=bound,raw=raw,uniform=uniform,radius=radius,distance=d,slope=slope)

def pack(result):
 return {k:serial(v) if isinstance(v,F) else v for k,v in result.items()}
