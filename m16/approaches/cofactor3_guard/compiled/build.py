"""Build once without executing the kernel. Retain failed build artifacts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build1'
assert not BUILD.exists(), 'Preserve any earlier build attempt'
BUILD.mkdir()
compiler = str(Path(shutil.which('cc')).resolve())
flags = ['-O2','-std=c11','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',
         '-frounding-math','-fexcess-precision=standard','-fno-tree-vectorize',
         '-fno-tree-slp-vectorize','-msse2','-mfpmath=sse']
command = [compiler, *flags, str(HERE/'guard.c'), '-lm', '-o', str(BUILD/'guard.so')]
record = {'compiler':compiler,'compiler_sha256':hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
          'version':subprocess.check_output([compiler,'--version'],text=True), 'command':command,
          'source_sha256':hashlib.sha256((HERE/'guard.c').read_bytes()).hexdigest()}
(BUILD/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
r = subprocess.run(command, capture_output=True,text=True)
(BUILD/'compile.stdout').write_text(r.stdout);(BUILD/'compile.stderr').write_text(r.stderr)
assert r.returncode == 0, r.returncode
for tool,args,dest in [('objdump',['-d','-M','intel'],'disassembly.txt'),
                       ('readelf',['-d'],'dynamic.txt')]:
    (BUILD/dest).write_bytes(subprocess.check_output([tool,*args,str(BUILD/'guard.so')]))
record['library_sha256'] = hashlib.sha256((BUILD/'guard.so').read_bytes()).hexdigest()
record['disassembly_sha256'] = hashlib.sha256((BUILD/'disassembly.txt').read_bytes()).hexdigest()
(BUILD/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
