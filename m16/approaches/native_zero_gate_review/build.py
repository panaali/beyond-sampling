import hashlib,json,resource,subprocess,time
from pathlib import Path
import prepare
P=Path(__file__).resolve().parent;(P/'data').mkdir(exist_ok=True);rows=[]
for name in ['original','fixed']:
    cmd=['g++','-std=c++17','-O3','-ffp-contract=off','-fPIC','-shared',str(P/(name+'.cpp')),'-o',str(P/'data'/(name+'.so')),'-l:liblapack.so.3','-l:libblas.so.3']
    before=resource.getrusage(resource.RUSAGE_CHILDREN);tick=time.perf_counter();subprocess.run(cmd,check=True);after=resource.getrusage(resource.RUSAGE_CHILDREN)
    rows.append(dict(name=name,command=cmd,wall_seconds=time.perf_counter()-tick,cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        source_sha256=hashlib.sha256((P/(name+'.cpp')).read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((P/'data'/(name+'.so')).read_bytes()).hexdigest()))
(P/'build.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
