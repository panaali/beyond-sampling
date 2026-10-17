import time
START=time.process_time()
import hashlib,json,resource
from pathlib import Path
import integrate as I
P=Path(__file__).resolve().parent
old=json.loads((P/'integration.json').read_text())['runs'][1]
new=I.integrate(I.cells(),16,radialorder=64,targetorder=64)
diffs={k:new['bylabel'][int(k)]-v for k,v in old['bylabel'].items()}
result=dict(original=old,refined=new,per_label_differences=diffs,source_sha256=hashlib.sha256((P/'integrate.py').read_bytes()).hexdigest(),counts=I.COUNT,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
assert max(abs(v) for v in diffs.values())<1e-10 and result['cpu_seconds_including_imports']<20
(P/'precision.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
