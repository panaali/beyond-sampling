"""Read-only map of pinned OpenBLAS runtime dispatch table to local ELF symbols."""
import ctypes,hashlib,json,re,subprocess
from pathlib import Path
from .inventory import HERE,LIB

def main():
    env=json.loads((HERE/'environment.json').read_text());assert hashlib.sha256(LIB.read_bytes()).hexdigest()==env['binary_hashes']['openblas'];lib=ctypes.CDLL(str(LIB));nm=subprocess.check_output(['nm','-n',str(LIB)],text=True);symbols={}
    for line in nm.splitlines():
        parts=line.split()
        if len(parts)==3:
            try:offset=int(parts[0],16)
            except ValueError:continue
            symbols[parts[2]]=offset
    base=ctypes.cast(lib.scipy_dgesv_64_,ctypes.c_void_p).value-symbols['scipy_dgesv_64_'];table=ctypes.c_void_p.from_address(base+symbols['gotoblas']).value;assert table and base<table<base+symbols['gotoblas']+10000000
    offsets=set()
    for name in ('dgetf2_k','dtrsm_LNLU','dtrsm_LNUN'):
        text=(HERE/(name+'_disassembly.txt')).read_text();offsets.update(int(x,16) for x in re.findall(r'call\s+\*0x([0-9a-f]+)\(',text))
    rows=[];byoffset={v:k for k,v in symbols.items()}
    for offset in sorted(offsets):
        target=ctypes.c_void_p.from_address(table+offset).value-base;assert target in byoffset;name=byoffset[target];text=subprocess.check_output(['objdump','-d','--disassemble='+name,str(LIB)],text=True);(HERE/(name+'_disassembly.txt')).write_text(text);rows.append(dict(table_offset=hex(offset),target_symbol=name,target_relative_elf_offset=hex(target),disassembly_sha256=hashlib.sha256(text.encode()).hexdigest()))
    constants={hex(k):ctypes.c_int.from_address(table+k).value for k in (0x2f0,0x2f4,0x2f8,0x2fc,0x300)}
    out=dict(read_only_own_process=True,binary_sha256=env['binary_hashes']['openblas'],table_relative_elf_offset=hex(table-base),block_parameter_offsets=constants,functions=rows,no_arithmetic_bound_asserted=True);(HERE/'runtime_dispatch.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
