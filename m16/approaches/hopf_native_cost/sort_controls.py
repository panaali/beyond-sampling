"""Frozen source-shadow stable sort and binary-search controls; no model runs."""
import ctypes,hashlib,itertools,json,math,time,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
 dst=HERE/'sort_controls.json';assert not dst.exists();start=time.process_time()
 out={'status':'running','cases':[],'search_cases':[]};save=lambda:dst.write_text(json.dumps(out,indent=2)+'\n');save()
 try:
  built=subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared',str(HERE/'sort_shadow.cpp'),'-o',str(HERE/'sort_shadow.so')],capture_output=True,text=True)
  out['build']={'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr};save();assert built.returncode==0
  lib=ctypes.CDLL(str(HERE/'sort_shadow.so'))
  for name,ctype in [('stable_double',ctypes.c_double),('stable_int64',ctypes.c_int64)]:
   fn=getattr(lib,name);fn.argtypes=[ctypes.POINTER(ctype),ctypes.POINTER(ctypes.c_int64),ctypes.c_int64,ctypes.POINTER(ctypes.c_uint64)];fn.restype=ctypes.c_int
  def shadow(values,indices=None):
   values=np.ascontiguousarray(values);idx=np.arange(len(values),dtype=np.int64) if indices is None else np.array(indices,dtype=np.int64,copy=True)
   counts=np.zeros(2,dtype=np.uint64);ctype=ctypes.c_double if values.dtype==np.float64 else ctypes.c_int64;fn=lib.stable_double if ctype is ctypes.c_double else lib.stable_int64
   ret=fn(values.ctypes.data_as(ctypes.POINTER(ctype)),idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),len(values),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
   assert ret==0
   assert counts[0]<=counts[1]<=counts[0]*(3 if ctype is ctypes.c_double else 1)
   return idx,counts.astype(int).tolist()
  def check(values,name):
   a=np.array(values);expected=np.argsort(a,kind='stable');got,c=shadow(a);assert np.array_equal(expected,got),(name,a,expected,got)
   out['cases'].append({'name':name,'n':len(a),'dtype':str(a.dtype),'calls':c[0],'scalar_comparisons':c[1],'permutation_sha256':hashlib.sha256(got.tobytes()).hexdigest()})
  for n in range(7):
   for i,p in enumerate(itertools.permutations(range(n))):
    for dtype in (np.float64,np.int64):check(np.array(p,dtype=dtype),f'permutation_{n}_{i}')
  for n in (15,16,31,32,63,64,65,127,128,257,1024):
   x=np.arange(n,dtype=np.int64)
   for kind,a in [('increasing',x),('reversed',x[::-1]),('duplicate',x%5),('modular',(37*x+11)%max(n,1))]:
    for dtype in (np.float64,np.int64):check(a.astype(dtype),f'{kind}_{n}')
   angles=((17*x+3)%13).astype(float);ids=(x%7).astype(np.int64)
   first,c1=shadow(angles);second,c2=shadow(ids,first);expected=np.lexsort((angles,ids));assert np.array_equal(second,expected)
   out['cases'].append({'name':f'lexsort_{n}','n':n,'calls':c1[0]+c2[0],'scalar_comparisons':c1[1]+c2[1],'permutation_sha256':hashlib.sha256(second.tobytes()).hexdigest()})
  for i,a in enumerate(([0.,-0.,float('nan'),-float('inf'),float('inf'),1.,-1.],[float('nan')]*5,[0.,-0.,0.,-0.])):check(np.array(a),f'nonfinite_{i}')
  def less(a,b):return a<b or (b!=b and a==a)
  for n in list(range(33))+[63,64,65,127,128,257,1024]:
   arr=np.sort(((np.arange(n)*17)%31).astype(float));keys=np.array([-float('inf'),-1.,0.,.5,4.,15.,30.,31.,float('inf'),float('nan')]);indices=[];calls=0
   for key in keys:
    if not n:indices.append(0);continue
    length=n;half=length>>1;length-=half;base=(not less(key,arr[half]))*half;calls+=1
    while length>1:
     half=length>>1;length-=half;base+=(not less(key,arr[base+half]))*half;calls+=1
    base+=not less(key,arr[base]);calls+=1;indices.append(base)
   expected=np.searchsorted(arr,keys,side='right');assert np.array_equal(indices,expected)
   predicted=len(keys)*(max((n-1).bit_length(),1)+1) if n else 0;assert calls==predicted
   out['search_cases'].append({'n':n,'keys':len(keys),'comparator_calls':calls,'indices':indices})
  out.update(status='complete',cpu_seconds=time.process_time()-start,cpp_sha256=hashlib.sha256((HERE/'sort_shadow.cpp').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Saved source-shadow comparison census only; native outputs unchanged, not installed-binary equivalence or control/memory price.');save();print(json.dumps({'status':out['status'],'sort_cases':len(out['cases']),'search_cases':len(out['search_cases']),'cpu_seconds':out['cpu_seconds']}))
 except Exception as e:out.update(status='failed',failure=repr(e),cpu_seconds=time.process_time()-start);save();raise
if __name__=='__main__':main()
