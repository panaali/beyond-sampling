"""Bounded read-only pointer metadata with checked mapping and durable steps."""
from pathlib import Path
import ctypes as C,hashlib,json,struct
import numpy as np
NP=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so')
BL=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
HERE=Path(__file__).resolve().parent
OUT=HERE/'dispatch_offset0.jsonl'
def record(d):
 with OUT.open('a') as f:f.write(json.dumps(d)+'\n');f.flush()
def maps():
 rows=[]
 for s in Path('/proc/self/maps').read_text().splitlines():
  x=s.split();a,b=map(lambda v:int(v,16),x[0].split('-'));rows.append((a,b,x[1],int(x[2],16),x[-1]))
 return rows
def read(addr,n):
 ranges=[r for r in maps() if r[0]<=addr and addr+n<=r[1] and r[2][0]=='r']
 record(dict(stage='read_check',address=hex(addr),n=n,mapped=bool(ranges)))
 assert ranges,'unmapped/unknown pointer: refusing read'
 return C.string_at(addr,n)
def base(p):
 rows=[r for r in maps() if r[-1]==str(p)];record(dict(stage='library_maps',path=str(p),rows=rows));assert rows
 zero=[r for r in rows if r[3]==0];assert len(zero)==1;return zero[0][0]
def main():
 assert not OUT.exists()
 nb=base(NP);bb=base(BL);get=C.pythonapi.PyCapsule_GetPointer;get.argtypes=[C.py_object,C.c_char_p];get.restype=C.c_void_p
 for stride in [8,24]:
  dt,cap=np.add._resolve_dtypes_and_context((np.dtype('d'),np.dtype('d'),None),reduction=True)
  np.add._get_strided_loop(cap,fixed_strides=(0,stride,0));addr=get(cap,b'numpy_1.24_ufunc_call_info')
  loop,ctx,aux=struct.unpack('QQQ',read(addr,24));record(dict(stage='info',stride=stride,loop=hex(loop),loop_offset=hex(loop-nb),context=hex(ctx),aux=hex(aux)))
  raw=struct.unpack('Q'*7,read(aux,56));record(dict(stage='aux',raw=[hex(x) for x in raw],legacy_candidate_offset=hex(raw[4]-nb)))
 got=struct.unpack('Q',read(bb+0x15ba770,8))[0];record(dict(stage='gotoblas',value=hex(got)))
 dot=struct.unpack('Q',read(got+0x368,8))[0];record(dict(stage='dot',address=hex(dot),offset=hex(dot-bb)))
 record(dict(stage='complete',binary_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [NP,BL]},scope='Current metadata only; no kernel or historical binding claim.'))
if __name__=='__main__':main()
