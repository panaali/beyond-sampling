import time
START=time.process_time()
import hashlib,json,resource,warnings
from pathlib import Path
warnings.filterwarnings('ignore',message='Unrecognized options detected')
import certify as C
P=Path(__file__).resolve().parent
cases=[('orthant',[[1,0],[0,1]],'feasible'),('opposed',[[1,0],[-1,0]],'infeasible'),('three_plane',[[1,0],[0,1],[-1,-1]],'infeasible'),('thin',[[1,0],[-1000000000000,1]],'feasible'),('strict_zero',[[1,0],[0,0]],'infeasible')]
results=[]
for name,rows,expected in cases:
    for scale in (1,10**12):
        scaled=[[v*scale for v in row] for row in rows];result=C.feasible(scaled)
        if result['status']!=expected or not C.verify(scaled,result):
            (P/'controls_failure.json').write_text(json.dumps(dict(name=name,scale=str(scale),rows=scaled,result=result,expected=expected,cpu_seconds=time.process_time()-START),indent=2)+'\n')
            raise AssertionError((name,scale,result))
        results.append(dict(name=name,scale=str(scale),rows=scaled,result=result))
output=dict(complete=True,controls=results,lp_calls=C.CALLS,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=hashlib.sha256((P/'certify.py').read_bytes()).hexdigest())
(P/'controls.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='controls'},indent=2))
