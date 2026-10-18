"""Read-only own-process norm dispatch, following reviewed T150 address method."""
import ctypes as ct,hashlib,json,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
LIB=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96'
lib=ct.CDLL(str(LIB));raw=subprocess.check_output(['nm','-n',str(LIB)],text=True);symbols={}
for line in raw.splitlines():
 p=line.split()
 if len(p)==3:
  try:symbols[p[2]]=int(p[0],16)
  except ValueError:pass
base=ct.cast(lib.scipy_dnrm2_64_,ct.c_void_p).value-symbols['scipy_dnrm2_64_'];table=ct.c_void_p.from_address(base+symbols['gotoblas']).value
assert table and base<table<base+symbols['gotoblas']+10000000
target=ct.c_void_p.from_address(table+0x348).value-base;names=[k for k,v in symbols.items() if v==target];assert names
name=names[0];f=lib.scipy_openblas_get_corename64_;f.restype=ct.c_char_p;core=f().decode();rows=[]
for n in ['scipy_dnrm2_64_',name]:
 text=subprocess.check_output(['objdump','-d','--disassemble='+n,str(LIB)],text=True);(HERE/(n+'.disasm.txt')).write_text(text);rows.append(dict(symbol=n,sha256=hashlib.sha256(text.encode()).hexdigest()))
out=dict(scope='8c9d87a4; no numeric bound yet',numpy=np.__version__,core=core,slot='0x348',target=name,target_elf_offset=hex(target),table_elf_offset=hex(table-base),disassembly=rows,reference_method='T150 inverse4_cost_review/dispatch_map.py; own-process reads only')
(HERE/'dispatch.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
