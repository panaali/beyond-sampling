"""Read current loop/BLAS dispatch metadata; no model or numeric kernel calls."""
from pathlib import Path
import ctypes as C,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
NP=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so')
BL=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
def base(p):
 rows=[s.split() for s in Path('/proc/self/maps').read_text().splitlines() if s.endswith(str(p))]
 assert rows;return min(int(r[0].split('-')[0],16)-int(r[2],16) for r in rows)
class Info(C.Structure):
 _fields_=[('loop',C.c_void_p),('context',C.c_void_p),('aux',C.c_void_p),('requires',C.c_byte),('no_fperr',C.c_byte)]
class Legacy(C.Structure):
 _fields_=[('free',C.c_void_p),('clone',C.c_void_p),('reserved',C.c_void_p*2),('loop',C.c_void_p),('user',C.c_void_p),('error',C.c_int)]
def main():
 out=HERE/'dispatch.json';assert not out.exists()
 nb=base(NP);bb=base(BL);get=C.pythonapi.PyCapsule_GetPointer;get.argtypes=[C.py_object,C.c_char_p];get.restype=C.c_void_p
 rows=[]
 for stride in [8,24]:
  dtypes,cap=np.add._resolve_dtypes_and_context((np.dtype('d'),np.dtype('d'),None),reduction=True)
  np.add._get_strided_loop(cap,fixed_strides=(0,stride,0))
  info=Info.from_address(get(cap,b'numpy_1.24_ufunc_call_info'));legacy=Legacy.from_address(info.aux)
  offset=legacy.loop-nb;assert offset in [0x5462e0,0x356fa0]
  rows.append(dict(stride=stride,adapter_offset=hex(info.loop-nb),legacy_offset=hex(offset),error_check=legacy.error,requires_api=info.requires,aux_loop_offset=Legacy.loop.offset))
 got=C.c_void_p.from_address(bb+0x15ba770).value;assert got
 dot=C.c_void_p.from_address(got+0x368).value;assert dot-bb==0xa3d7c0
 result=dict(rows=rows,dot_offset=hex(dot-bb),gotoblas_pointer_slot='0x368',binary_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [NP,BL]},scope='Current process metadata and pinned ABI only; not proof of historical binding or complete reduction chunk trace. No numeric kernels called.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
