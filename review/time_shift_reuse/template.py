"""Exact conditional gate-word algebra; not a cone/probability cache."""
from fractions import Fraction as F
from collections import Counter

class Arithmetic:
 def __init__(self):self.count=Counter();self.bits=[0,0]
 def op(self,k,a,b):
  self.count[k]+=1;v=a*b if k=='mul' else a+b
  self.bits=[max(self.bits[0],abs(v.numerator).bit_length()),max(self.bits[1],v.denominator.bit_length())];return v
 def dot(self,a,b):
  v=self.op('mul',a[0],b[0])
  for x,y in zip(a[1:],b[1:],strict=True):v=self.op('add',v,self.op('mul',x,y))
  return v

def build(W,w,word):
 H=len(w);k=len(word);d=H+k;C=Arithmetic()
 assert H>0 and k>0 and len(W)==H and all(len(r)==H for r in W)
 assert all(len(g)==H and all(type(x) is int and x in (0,1) for x in g) for g in word)
 A=tuple(tuple(F(i==j) for j in range(d)) for i in range(H));planes=[]
 for t,g in enumerate(word):
  B=tuple(tuple(C.op('add',C.dot(W[i],tuple(A[j][q] for j in range(H))),w[i] if q==H+t else F(0)) for q in range(d)) for i in range(H))
  planes.extend((B[i],g[i]) for i in range(H))
  A=tuple(B[i] if g[i] else (F(0),)*d for i in range(H))
 return dict(H=H,length=k,output=A,planes=tuple(planes),impossible_identically_zero_on=any(on and not any(row) for row,on in planes),counts=dict(C.count),bits=C.bits)

def compose(left,right):
 H=left['H'];assert H==right['H'];a=left['length'];b=right['length'];d=H+a+b;C=Arithmetic()
 transfer=tuple(tuple(row)+(F(0),)*b for row in left['output'])+tuple(tuple(F(q==H+a+j) for q in range(d)) for j in range(b))
 def transform(row):return tuple(C.dot(row,tuple(transfer[i][q] for i in range(H+b))) for q in range(d))
 output=tuple(transform(row) for row in right['output'])
 planes=tuple((row+(F(0),)*b,on) for row,on in left['planes'])+tuple((transform(row),on) for row,on in right['planes'])
 return dict(H=H,length=a+b,output=output,planes=planes,impossible_identically_zero_on=any(on and not any(row) for row,on in planes),counts=dict(C.count),bits=C.bits)

def shifted(template,start,ambient_inputs):
 H=template['H'];k=template['length'];assert start>=0 and start+k<=ambient_inputs
 def insert(row):
  out=[F(0)]*(H+ambient_inputs);out[:H]=row[:H];out[H+start:H+start+k]=row[H:];return tuple(out)
 return dict(output=tuple(insert(r) for r in template['output']),planes=tuple((insert(r),on) for r,on in template['planes']))

def pack(x):
 if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
 if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [pack(v) for v in x]
 return x
