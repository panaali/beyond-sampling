"""Fixed synthetic passthrough controls; no model or estimator imports."""
from pathlib import Path
import ctypes as C,json,hashlib,struct,math,time,resource,traceback
import numpy as np
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def maps():
 out=[]
 for line in Path('/proc/self/maps').read_text().splitlines():
  x=line.split();a,b=(int(v,16) for v in x[0].split('-'));out.append((a,b,x[1],int(x[2],16),x[-1]))
 return out

def mapped(addr,n,write=False):
 valid=[r for r in maps() if r[0]<=addr and addr+n<=r[1] and r[2][0]=='r' and (not write or r[2][1]=='w')]
 assert valid,('invalid mapped pointer',hex(addr),n,write);return valid[0]
def pointer(addr):mapped(addr,8);return struct.unpack('Q',C.string_at(addr,8))[0]
def base(path):
 rows=[r for r in maps() if r[-1]==str(path) and r[3]==0];assert len(rows)==1;return rows[0][0]
def bits(x):return struct.pack('d',float(x)).hex()
def layout(values,kind):
 a=np.array(values,dtype=np.float64)
 if kind=='contiguous':return a
 if kind=='stride2':
  b=np.full(6,123.0,dtype=np.float64);b[::2]=a;return b[::2]
 if kind=='reversed':return a[::-1]
 raise ValueError(kind)
def apply(kind,a,b):
 if kind=='norm':return np.linalg.norm(a)
 if kind=='dot':return np.dot(a,b)
 if kind=='matmul':return a@b
 raise ValueError(kind)

def main():
 out=P/'results.json';assert not out.exists(),'immutable attempt exists'
 protocol=json.loads((P/'protocol.json').read_text());build=json.loads((P/'build.json').read_text());assert sha(P/'spy.so')==build['binary_sha256'] and sha(P/'spy.c')==build['source_sha256']
 for path,digest in protocol['pins'].items():assert sha(path)==digest,path
 assert np.__version__==protocol['numpy_version']
 start=time.process_time();d=dict(complete=False,protocol_sha256=sha(P/'protocol.json'),source_sha256=sha(__file__),spy_build=build,rows=[])
 def save():
  d['cpu']=time.process_time()-start;d['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;out.write_text(json.dumps(d,indent=2)+'\n')
 save();spy=None;installed=False
 try:
  lib=C.CDLL(None);lib.fegetround.restype=C.c_int;assert lib.fegetround()==0,'frozen round-to-nearest environment required'
  bl=Path(protocol['blas_library']);nplib=Path(protocol['numpy_library']);bb=base(bl);nb=base(nplib)
  table=pointer(bb+0x15ba770);mapped(table+0x368,8,True);target=pointer(table+0x368);mapped(target,32);assert target-bb==0xa3d7c0
  d['runtime']=dict(blas_base=hex(bb),numpy_base=hex(nb),table=hex(table),dot_target=hex(target),dot_offset=hex(target-bb),slot_offset='0x368',rounding=lib.fegetround(),library_maps=[r for r in maps() if r[-1] in (str(bl),str(nplib))]);save()
  spy=C.CDLL(str(P/'spy.so'));spy.install.argtypes=[C.c_void_p,C.c_void_p];spy.install.restype=C.c_int;spy.restore.restype=C.c_int;spy.reset.restype=None;spy.event_count.restype=C.c_long;spy.event_overflow.restype=C.c_int;spy.event_value.argtypes=[C.c_long,C.c_long];spy.event_value.restype=C.c_long
  fixtures=[]
  with np.errstate(all='ignore'):
   for pi,pattern in enumerate(protocol['patterns_hex']):
    values=[float.fromhex(v) for v in pattern];other=[values[2],values[0],values[1]]
    for lo in protocol['layouts']:
     a=layout(values,lo);b=layout(other,lo)
     assert a.dtype==b.dtype==np.float64 and a.shape==b.shape==(3,)
     for op in protocol['operations']:
      baseline=apply(op,a,b);fixtures.append((a,b,op,dict(pattern=pi,layout=lo,operation=op,input_a_hex=[float(x).hex() for x in a],input_b_hex=[float(x).hex() for x in b],strides_a=list(a.strides),strides_b=list(b.strides),aligned_a=bool(a.flags.aligned),aligned_b=bool(b.flags.aligned),baseline_bits=bits(baseline))))
   assert len(fixtures)==90
   assert spy.install(table,target)==1;installed=True
   for a,b,op,row in fixtures:
    spy.reset();value=apply(op,a,b);assert not spy.event_overflow();events=[dict(n=spy.event_value(i,0),incx=spy.event_value(i,1),incy=spy.event_value(i,2)) for i in range(spy.event_count())]
    row.update(value_bits=bits(value),events=events,parity=bits(value)==row['baseline_bits']);d['rows'].append(row);save();assert row['parity'],row
   assert spy.restore()==1;installed=False
  assert pointer(table+0x368)==target;assert lib.fegetround()==0
  d.update(complete=True,restored=True,control_count=len(d['rows']),all_bitwise_parity=all(r['parity'] for r in d['rows']),scope='Finite90 synthetic passthrough controls only. Recorded native event paths do not establish historical trained operand layouts or complete native/hardware costs. No count substitution.')
  save();print(json.dumps({k:d[k] for k in ('complete','control_count','all_bitwise_parity','cpu','max_rss_kib')}))
 except Exception:
  d['failure']=traceback.format_exc();save();raise
 finally:
  if installed:
   restored=bool(spy.restore());d['restored_after_failure']=restored;save();assert restored
if __name__=='__main__':main()
