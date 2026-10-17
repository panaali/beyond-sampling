import hashlib,json,resource,subprocess,time
from pathlib import Path
p=Path(__file__).parent;start=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
command=['g++','-std=c++17','-O3','-ffp-contract=off','-fPIC','-shared',str(p/'projective.cpp'),'-o',str(p/'data/projective.so')]
subprocess.run(command,check=True);after=resource.getrusage(resource.RUSAGE_CHILDREN)
report=dict(command=command,wall_seconds=time.perf_counter()-start,cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
 source_sha256=hashlib.sha256((p/'projective.cpp').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((p/'data/projective.so').read_bytes()).hexdigest())
(p/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
