"""Compare all frozen analytic arms against already saved guarded results."""
import hashlib,json,resource,time
import numpy as np
from observer import HERE,load,install
FX,F3,Ops=load();obs,guard=install(FX,F3);rows=[];start=time.process_time()
for name in ('zero','constant0','plus3_minus0','plus0_minus3'):
 old=json.loads((HERE/('guard_'+name+'.json')).read_text());model=tuple(np.array(x) for x in old['model'])
 for arm in old['arms']:
  ops=Ops();stats={};tier=arm['tier'];t=time.process_time()
  got=F3.est_F2L2_4(model,ops,stats) if tier=='F2L3' else FX.est_fibre4(model,'FX',ops,stats)
  row={'case':name,'tier':tier,'estimate':got,'legacy_ops':ops.n,'stats':stats,'cpu_seconds':time.process_time()-t}
  assert got==arm['estimate'] and ops.n==arm['legacy_ops'] and stats==arm['stats'],row
  rows.append(row)
result={'rows':rows,'observer':obs.counts,'guard':guard,'cpu_seconds':time.process_time()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'observer_sha256':hashlib.sha256((HERE/'observer.py').read_bytes()).hexdigest()}
(HERE/'observer_controls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
