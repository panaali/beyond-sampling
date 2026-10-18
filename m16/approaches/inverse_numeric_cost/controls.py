"""Fixed alphabet binary IAMAX branch check; fixed inverse passthrough controls."""
from pathlib import Path
import ctypes,hashlib,itertools,json,math,resource,subprocess,time
import numpy as np
P=Path(__file__).resolve().parent
OLD=P.parent/'inverse4_cost_review'

def model(x):
 # MAXPD/MAXSD choose source operand when equal or unordered.
 def mx(a,b):return a if a>b else b
 n=len(x);r=n-1;ab=2;cmp=0
 s=[abs(x[0])]*2;v0=s.copy();v1=s.copy();v2=s.copy();v3=s.copy()
 if r&2:
  v2=[mx(v2[k],abs(x[1+k])) for k in range(2)];ab+=2;cmp+=2
 if r&1:
  z=abs(x[-1]);v3=[mx(v,z) for v in v3];ab+=2;cmp+=2
 v0=[mx(a,b) for a,b in zip(v0,v1)];v2=[mx(a,b) for a,b in zip(v2,v3)];v0=[mx(a,b) for a,b in zip(v0,v2)];y=mx(v0[1],v0[0]);cmp+=7
 # Assembly loads all entries of a 4- or 2-entry scan block before comparing.
 scanned=4 if n==4 else (2 if n>=2 else 0)
 ab+=2*scanned
 for j in range(scanned):
  cmp+=1;z=abs(x[j])
  if z==y or math.isnan(z) or math.isnan(y):return j+1,ab,cmp
 return scanned+1,ab,cmp

def outcome(A):
 try:
  with np.errstate(all='ignore'):v=np.linalg.inv(A)
  return {'value_hex':[float(x).hex() for x in v.ravel()]}
 except np.linalg.LinAlgError as e:return {'failure':type(e).__name__,'message':str(e)}

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
 env=json.loads((OLD/'environment.json').read_text());libpath=Path(env['binary_paths']['openblas']);assert hashlib.sha256(libpath.read_bytes()).hexdigest()==env['binary_hashes']['openblas']
 lib=ctypes.CDLL(str(libpath));raw=subprocess.check_output(['nm','-n',str(libpath)],text=True);symbols={z[2]:int(z[0],16) for l in raw.splitlines() if len(z:=l.split())==3 and all(c in '0123456789abcdef' for c in z[0])}
 base=ctypes.cast(lib.scipy_dgesv_64_,ctypes.c_void_p).value-symbols['scipy_dgesv_64_'];table=ctypes.c_void_p.from_address(base+symbols['gotoblas']).value
 addr=ctypes.c_void_p.from_address(table+0x328).value;assert addr-base==symbols['idamax_k_SKYLAKEX']
 iamax=ctypes.CFUNCTYPE(ctypes.c_int64,ctypes.c_int64,ctypes.POINTER(ctypes.c_double),ctypes.c_int64)(addr)
 alphabet=[-2.,-0.,0.,1.,float('inf'),float('-inf'),float('nan'),2.**-1074]
 digest=hashlib.sha256();counts=[]
 for n in range(1,5):
  maxabs=maxcmp=0;count=0
  for x in itertools.product(alphabet,repeat=n):
   expect,a,c=model(x);buf=(ctypes.c_double*n)(*x);got=iamax(n,buf,1)
   if got!=expect:
    (P/'failure.json').write_text(json.dumps(dict(x=[v.hex() for v in x],expected=expect,actual=got),indent=2)+'\n');raise AssertionError((x,expect,got))
   digest.update(json.dumps(([v.hex() for v in x],got,a,c)).encode());count+=1;maxabs=max(maxabs,a);maxcmp=max(maxcmp,c)
  counts.append(dict(n=n,cases=count,max_abs_lanes=maxabs,max_cmp_lanes=maxcmp))
 # Baseline first, then only local process dispatch pointers replaced transparently.
 D=np.array([[4.,1,2,0],[1,5,0,1],[2,0,6,1],[0,1,1,3]])
 fixtures=[('identity',np.eye(4)),('row_pivot',D[[3,1,0,2]]),('zero',np.zeros((4,4))),('duplicate_row',D[[0,1,1,3]]),('subnormal',np.diag([2.**-1023,1,1,1])),('nan_diagonal',np.diag([float('nan'),1,1,1])),('infinity_diagonal',np.diag([float('inf'),1,1,1])),('finite_overflow',D*2.**1020)]
 baseline={name:outcome(A) for name,A in fixtures}
 spy=ctypes.CDLL(str(OLD/'kernel_spy.so'));spy.kernel_install.argtypes=[ctypes.c_size_t];spy.kernel_events.argtypes=[ctypes.POINTER(ctypes.c_int64)];spy.kernel_events.restype=ctypes.c_int64;names=['iamax','dot','scal','swap','gemv','gemm','copy','upper','lower','upack','lpack'];rows=[]
 spy.kernel_install(table)
 try:
  for name,A in fixtures:
   spy.kernel_reset();got=outcome(A);assert got==baseline[name],name
   buf=(ctypes.c_int64*(256*5))();n=spy.kernel_events(buf);assert 0<=n<256
   events=[dict(kernel=names[buf[5*i]],args=list(buf[5*i+1:5*i+5])) for i in range(n)]
   assert all(e['kernel']!='gemm' for e in events)
   assert [e['args'][0] for e in events if e['kernel']=='iamax']==[4,3,2,1]
   rows.append(dict(name=name,input_hex=[float(v).hex() for v in A.ravel()],**got,events=events))
 finally:spy.kernel_restore()
 assert ctypes.c_void_p.from_address(table+0x328).value==addr
 out=dict(binary_hash=env['binary_hashes']['openblas'],alphabet=[v.hex() for v in alphabet],iamax_cases=counts,iamax_result_sha256=digest.hexdigest(),inverse_controls=rows,baseline_parity=True,dispatch_restored=True,cpu=time.process_time()-start,complete=True)
 (P/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('inverse_controls','alphabet')}))
if __name__=='__main__':main()
