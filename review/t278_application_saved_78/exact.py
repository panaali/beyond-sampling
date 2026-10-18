"""Independent exact feasible-mask reconstruction and interval-set arithmetic."""
from fractions import Fraction as F
from itertools import product

def decode(x):
 if isinstance(x,dict):
  if set(x)=={'q'}:
   a=x['q'];assert isinstance(a,list) and len(a)==2 and all(type(v)is str and str(int(v))==v for v in a) and int(a[1])>0
   v=F(int(a[0]),int(a[1]));assert a==[str(v.numerator),str(v.denominator)];return v
  return {k:decode(v) for k,v in x.items()}
 if isinstance(x,list):return [decode(v) for v in x]
 return x

def reconstruct(model,y,direction,width):
 wi,wh,wo=model;H=len(wi);states=[(-width,width,[(F(0),F(0))]*H)];layers=[];assignments=0
 for t in range(len(y)):
  nextstates=[]
  for lo,hi,h in states:
   pre=[(wi[i]*y[t]+sum(wh[i][j]*h[j][0] for j in range(H)),wi[i]*direction[t]+sum(wh[i][j]*h[j][1] for j in range(H))) for i in range(H)]
   for bits in product((0,1),repeat=H):
    assignments+=1;a,b=lo,hi;valid=True
    for on,(c,s) in zip(bits,pre):
     if not s:
      if bool(on)!=(c>0):valid=False;break
     else:
      root=-c/s
      if (s>0)==bool(on):a=max(a,root)
      else:b=min(b,root)
    if valid and a<b:nextstates.append((a,b,[pre[i] if bits[i] else (F(0),F(0)) for i in range(H)]))
  states=sorted(nextstates);assert states and states[0][0]==-width and states[-1][1]==width
  assert all(a[1]==b[0] for a,b in zip(states,states[1:]));layers.append(len(states))
 def outputs(h):return [[sum(row[j]*h[j][k] for j in range(H)) for k in range(2)] for row in wo]
 actual=[[a,b,outputs(h)] for a,b,h in states]
 h=[(F(0),F(0))]*H
 for t in range(len(y)):
  pre=[(wi[i]*y[t]+sum(wh[i][j]*h[j][0] for j in range(H)),wi[i]*direction[t]+sum(wh[i][j]*h[j][1] for j in range(H))) for i in range(H)]
  h=[(c,s) if c>0 else (F(0),F(0)) for c,s in pre]
 return actual,layers,[[-width,width,outputs(h)]],[1]*len(y),assignments

def merged(parts):
 out=[]
 for a,b in sorted(parts):
  assert a<b
  if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
  else:out.append([a,b])
 return out

def errors(cells,truth,lo,hi):
 out=[]
 for left,right,ls in cells:
  a,b=max(left,lo),min(right,hi)
  if a>=b:continue
  cuts={a,b}
  for i in range(len(ls)):
   for j in range(i):
    c=ls[i][0]-ls[j][0];s=ls[i][1]-ls[j][1]
    if s and a<-c/s<b:cuts.add(-c/s)
  cuts=sorted(cuts)
  for l,u in zip(cuts,cuts[1:]):
   mid=(l+u)/2;winner=max(range(len(ls)),key=lambda j:ls[j][0]+ls[j][1]*mid)
   if winner!=truth:out.append([l,u])
 return merged(out)

def difference(actual,frozen):
 jumps={}
 for s,parts in ((1,actual),(-1,frozen)):
  for a,b in parts:jumps[a]=jumps.get(a,0)+s;jumps[b]=jumps.get(b,0)-s
 cuts=sorted(jumps);level=0;out=[]
 for a,b in zip(cuts,cuts[1:]):
  level+=jumps[a];assert level in (-1,0,1)
  if level:
   if out and out[-1][1]==a and out[-1][2]==level:out[-1][1]=b
   else:out.append([a,b,level])
 return out
