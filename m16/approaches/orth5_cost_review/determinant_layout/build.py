"""Build-only observer gate: no determinant or NumPy calls."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
assert not (HERE/'build.json').exists() and not (HERE/'spy.so').exists()
start=time.process_time()
compiler=shutil.which('gcc');assert compiler
flags=['-shared','-fPIC','-O2','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror']
command=[compiler,*flags,str(HERE/'spy.c'),'-ldl','-o',str(HERE/'spy.so')]
p=subprocess.run(command,capture_output=True,text=True)
out={'complete':p.returncode==0,'command':command,'compiler_version':subprocess.check_output([compiler,'--version'],text=True),
     'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,
     'source_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['spy.c','build.py','PROTOCOL.md']},
     'driver_cpu_seconds_excluding_compiler_child':time.process_time()-start,'numerical_calls':0}
if p.returncode==0:
    out['library_sha256']=hashlib.sha256((HERE/'spy.so').read_bytes()).hexdigest()
    dis=subprocess.check_output(['objdump','-d',str(HERE/'spy.so')],text=True)
    (HERE/'spy_disassembly.txt').write_text(dis)
    out['disassembly_sha256']=hashlib.sha256(dis.encode()).hexdigest()
(HERE/'build.json').write_text(json.dumps(out,indent=2)+'\n')
assert out['complete'],out
print(json.dumps({k:out[k] for k in ['complete','library_sha256','numerical_calls']}))
