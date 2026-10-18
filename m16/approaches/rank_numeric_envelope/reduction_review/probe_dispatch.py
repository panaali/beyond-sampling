"""Read-only own-process scale dispatch, following reviewed T150 address method."""
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
rows=[]
for slot in [0x388,0x390,0x3a0,0x3a8,0x3b0]:
 target=ct.c_void_p.from_address(table+slot).value-base
 names=[k for k,v in symbols.items() if v==target];assert names
 name=names[0]
 text=subprocess.check_output(['objdump','-d','--disassemble='+name,str(LIB)],text=True)
 (HERE/(name+'.disasm.txt')).write_text(text)
 rows.append(dict(slot=hex(slot),symbol=name,elf_offset=hex(target),sha256=hashlib.sha256(text.encode()).hexdigest()))
out=dict(numpy=np.__version__,table_elf_offset=hex(table-base),rows=rows,scope='Own-process pointer probe only; no numeric kernels executed; fixed historical binding remains separate')
assert not (HERE/'blas_dispatch.json').exists()
(HERE/'blas_dispatch.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
