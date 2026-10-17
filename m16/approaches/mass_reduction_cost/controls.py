"""Frozen source-operation parity controls; no estimator calls."""
from pathlib import Path
from fractions import Fraction
import ctypes as C,hashlib,itertools,json,math,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
class SavedRows(list):
 def __init__(self,path):
  super().__init__();assert not path.exists();self.path=path
 def append(self,row):
  super().append(row)
  with self.path.open('a') as f:f.write(json.dumps(row)+'\n')

def enc(x):return float(x).hex()
def same(a,b):return (math.isnan(float(a)) and math.isnan(float(b))) or enc(a)==enc(b)
def pair(a):
 n=len(a)
 if n<8:
  v=np.float64(-0.0)
  for x in a:v=np.float64(v+x)
  return v,n
 if n<=128:
  r=[np.float64(x) for x in a[:8]];k=8
  while k<n-n%8:
   for j in range(8):r[j]=np.float64(r[j]+a[k+j])
   k+=8
  v=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]))
  for x in a[k:]:v=np.float64(v+x)
  return v,n-1
 k=(n//2)//8*8;l,lc=pair(a[:k]);r,rc=pair(a[k:]);return np.float64(l+r),lc+rc+1

def sumref(a):
 v,n=pair(a);return np.float64(np.float64(0)+v),n+1

def pysumref(a):
 # Exactly six built-in floats; 0+first is the int-to-float transition.
 v=0+a[0];c=0.0;branches=[]
 for x in a[1:]:
  t=v+x;left=abs(v)>=abs(x)
  c+=(v-t)+x if left else (x-t)+v
  branches.append(left);v=t
 if c and math.isfinite(c):v+=c
 return v,branches,c

def fmaref(a):
 v=0.0
 for x in a:
  if not math.isfinite(float(x)) or not math.isfinite(v):
   v=float(np.float64(v)+np.float64(x)*np.float64(x));continue
  z=Fraction(v)+Fraction(float(x))*Fraction(float(x))
  try:v=float(z)
  except OverflowError:v=math.inf
 return float(np.float64(0)+np.float64(v))

def main():
 out=HERE/'controls.json';assert not out.exists()
 protocol=json.loads((HERE/'protocol.json').read_text());assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==protocol['controls_sha256']
 start=time.process_time();rows=SavedRows(HERE/'control_progress.jsonl');fail=[]
 patterns=[[0.0],[-0.0],[1.0,-1.0],[1e16,1,-1e16,1],[math.pi,-math.pi,2**-52,1],[1e-200,-1e-200],[1e100,-1e100,1],[float.fromhex('0x0.0000000000001p-1022'),-0.0],[math.inf,1], [math.nan,1]]
 with np.errstate(all='ignore'):
  for shape in [(4,),(7,),(8,),(128,),(1000,),(1000,4)]:
   for j,pat in enumerate(patterns):
    a=np.resize(np.array(pat,dtype=np.float64),shape);assert a.flags.c_contiguous
    actual=np.sum(a,axis=1) if a.ndim==2 else np.sum(a)
    aa=a if a.ndim==2 else a[None,:];refs=[sumref(x) for x in aa];expected=np.array([x[0] for x in refs]);av=np.atleast_1d(actual)
    ok=all(same(x,y) for x,y in zip(av,expected));row=dict(kind='numpy_sum',shape=shape,pattern=j,ok=ok,count=sum(x[1] for x in refs),values=[enc(x) for x in av],reference=[enc(x) for x in expected]);rows.append(row)
    if not ok:fail.append(len(rows)-1)
  for j,pat in enumerate(patterns):
   base=np.resize(np.array(pat,dtype=np.float64),4)
   for rot in range(4):
    a=np.roll(base,rot);expected=math.sqrt(fmaref(a));actual=np.linalg.norm(a);ok=same(actual,expected)
    rows.append(dict(kind='norm4',pattern=j,rotation=rot,input=[enc(x) for x in a],value=enc(actual),reference=enc(expected),ok=ok))
    if not ok:fail.append(len(rows)-1)
  bases=[list(map(float,np.resize(p,6))) for p in patterns]+[list(map(float,a)) for a in [[1e16,1,1,-1e16,1,1],[1.0,2**-53,2**-53,-1.0,2**-100,-2**-100]]]
  for j,base in enumerate(bases):
   for rot in range(6):
    a=base[rot:]+base[:rot];assert all(type(x) is float for x in a);expected,branches,c=pysumref(a);actual=sum(a);ok=same(actual,expected)
    rows.append(dict(kind='python_sum6',pattern=j,rotation=rot,input=[enc(x) for x in a],value=enc(actual),reference=enc(expected),branches=branches,compensation=enc(c),ok=ok))
    if not ok:fail.append(len(rows)-1)
 assert len(rows)==172
 out.write_text(json.dumps(dict(cases=len(rows),failures=fail,rows=rows,cpu_seconds=time.process_time()-start,scope='Finite source-order parity only. Norm uses independently rounded Fraction FMA, not original BLAS reference. NumPy pairwise/source and Python compensation translations validate values, not binding or instruction counts.'),indent=2)+'\n')
 print(json.dumps(dict(cases=len(rows),failures=fail)))
if __name__=='__main__':main()
