"""Read the already pinned runtime dispatch table; no matrix or mass call."""
import ctypes,hashlib,json,resource,subprocess,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();manifest=json.loads((HERE/'manifest.json').read_text());path=manifest['binary_paths']['openblas']
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest['binary_sha256']['openblas']
    lib=ctypes.CDLL(path);lib.scipy_openblas_get_num_threads64_.restype=ctypes.c_int;assert lib.scipy_openblas_get_num_threads64_()==1
    nm=subprocess.check_output(['nm','-n',path],text=True);syms={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3 and all(c in '0123456789abcdef' for c in p[0])};addresses={int(p[0],16):p[2] for line in nm.splitlines() if len(p:=line.split())==3 and p[1] in ('T','t')}
    base=ctypes.cast(lib.scipy_dgemm_64_,ctypes.c_void_p).value-syms['scipy_dgemm_64_'];table=ctypes.c_void_p.from_address(base+syms['gotoblas']).value
    # The pinned CBLAS entry disassembly accesses table offset0x17f8,
    # establishing this finite readable table prefix. Never dereference targets.
    rows=[]
    for off in range(0,0x1800,8):
        ptr=ctypes.c_void_p.from_address(table+off).value
        if ptr and ptr-base in addresses:rows.append(dict(offset=hex(off),symbol=addresses[ptr-base],target=hex(ptr-base)))
    out=dict(binary_sha256=manifest['binary_sha256']['openblas'],table_relative_offset=hex(table-base),read_only_prefix_bytes=0x1800,entries=rows,cpu_seconds=time.process_time()-start,new_matrix_mass_calls=0)
    (HERE/'runtime_table.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(entries=len(rows),cpu_seconds=out['cpu_seconds'],matrix_entries=[r for r in rows if any(x in r['symbol'] for x in ('dgemm','dsyrk'))])))
if __name__=='__main__':main()
