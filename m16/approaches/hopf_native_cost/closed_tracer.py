"""Attach separately qualified source-level sort/search accounting to original calls."""
import ctypes,json,hashlib
from pathlib import Path
import numpy as np
import tracer
HERE=Path(__file__).resolve().parent
class Ledger(tracer.Ledger):
 def __init__(self):
  super().__init__();self.closures=[];self.lib=ctypes.CDLL(str(HERE/'sort_shadow.so'))
  for name,ctype in [('stable_double',ctypes.c_double),('stable_int64',ctypes.c_int64)]:
   fn=getattr(self.lib,name);fn.argtypes=[ctypes.POINTER(ctype),ctypes.POINTER(ctypes.c_int64),ctypes.c_int64,ctypes.POINTER(ctypes.c_uint64)];fn.restype=ctypes.c_int
 def stable(self,values,indices):
  assert values.dtype in (np.dtype('float64'),np.dtype('int64')) and values.ndim==1
  arr=np.ascontiguousarray(values);idx=np.array(indices,dtype=np.int64,copy=True);counts=np.zeros(2,dtype=np.uint64)
  fp=arr.dtype==np.float64;ctype=ctypes.c_double if fp else ctypes.c_int64;fn=self.lib.stable_double if fp else self.lib.stable_int64
  ret=fn(arr.ctypes.data_as(ctypes.POINTER(ctype)),idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),len(arr),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)));assert ret==0
  assert counts[0]<=counts[1]<=counts[0]*(3 if fp else 1)
  return idx,int(counts[0]),int(counts[1])
 def call(self,site,name,fn,*args,**kwargs):
  result=super().call(site,name,fn,*args,**kwargs)
  if name=='np.lexsort':
   assert len(args)==1 and not kwargs and len(args[0])==2
   keys=args[0];assert keys[0].dtype==np.float64 and keys[1].dtype==np.int64
   assert all(v.ndim==1 and len(v)==len(keys[0]) for v in keys) and np.all(np.isfinite(keys[0]))
   indices=np.arange(len(keys[0]),dtype=np.int64);calls=comps=0
   for values in keys:
    indices,c,q=self.stable(values,indices);calls+=c;comps+=q
   assert np.array_equal(indices,result),'source-shadow lexsort permutation mismatch'
   self.closures.append({'site':site,'name':name,'n':len(result),'source_comparator_calls':calls,'source_scalar_comparisons':comps,'numerical_allowance':comps,'kind':'shadow source; binary equivalence qualified'})
  elif name=='np.searchsorted':
   assert len(args)==3 and args[2]=='right' and not kwargs
   a,v=args[:2];assert a.ndim==v.ndim==1 and a.dtype==v.dtype==np.float64 and np.all(np.isfinite(a)) and np.all(np.isfinite(v))
   n=len(a);calls=len(v)*(max((n-1).bit_length(),1)+1) if n else 0
   self.closures.append({'site':site,'name':name,'n':n,'keys':len(v),'source_comparator_calls':calls,'numerical_allowance':4*calls,'kind':'source upper:3 scalar comparisons plus Boolean inversion per right comparator'})
  elif name=='np.argsort':
   assert len(args)==2 and args[1]==1 and not kwargs
   a=args[0];assert a.ndim==2 and a.shape[1]==4 and a.dtype==np.float64 and np.all(np.isfinite(a))
   self.closures.append({'site':site,'name':name,'rows':len(a),'n':4,'numerical_allowance':157*len(a),'kind':'conditional finite source envelope; control/memory/native equivalence excluded'})
  return result
 def report(self):
  out=super().report();out['source_closures']=self.closures.copy();out['source_closure_allowance']=sum(x['numerical_allowance'] for x in self.closures)
  opaque=sum(x['count'] for x in out['opaque_calls']);assert opaque==len(self.closures),(opaque,len(self.closures))
  out['explicit_plus_conditional_source_allowance']=out['inclusive_explicit_total']+out['source_closure_allowance']
  out['complete_implementation_cost']=False
  return out

def load(rescue=False):
 mod,_,startup=tracer.load(True,rescue);ledger=Ledger();mod._arithmetic=ledger
 return mod,ledger,startup
