"""Fixed primitive clips/dense rows, then saved analytic candidate parity."""
import hashlib,json,resource,sys,time
import numpy as np
from tracer import HERE,load
PRIOR=HERE.parent/'fibre_t4_recovery'
name=sys.argv[1];assert name in ('primitive','zero','constant0','plus3_minus0','plus0_minus3')
path=HERE/('controls_'+name+'.json');assert not path.exists(),'Preserve attempts'
start=time.process_time();out={'control':name,'complete':False,'tracer_sha256':hashlib.sha256((HERE/'tracer.py').read_bytes()).hexdigest()};path.write_text(json.dumps(out,indent=2)+'\n')
try:
 if name=='primitive':
  # Convex spherical quadrilateral, with generic and EPS-near cut planes.
  poly=[np.array(v,dtype=float)/np.linalg.norm(v) for v in ((1,1,1),(-1,1,1),(-1,-1,1),(1,-1,1))]
  normals=[np.zeros(3),np.array([0.,0.,1.]),np.array([0.,0.,-1.]),np.array([1.,0.,0.]),np.array([1.,1+2e-11,0.]),np.array([1.,1-2e-11,0.]),np.array([1.,1+1e-10,0.]),np.array([1.,1-1e-10,0.])]
  saved=[]
  for enabled in (False,True):
   FX,F3,Ops,ledger=load(enabled);rows=[]
   for normal in normals:
    ops=Ops();a,b=FX.clip(poly,normal,ops);rows.append({'normal':normal.tolist(),'in':[v.tolist() for v in a],'out':[v.tolist() for v in b],'legacy_ops':ops.n})
   # Existing exact-sign models exercise dense arrays with zeros and active gates.
   for case in ('zero','constant0','plus3_minus0','plus0_minus3'):
    m=tuple(np.array(w) for w in json.loads((PRIOR/('guard_'+case+'.json')).read_text())['model'])
    ops=Ops();R,L,live=FX.rows_at(m,np.array([-2.,-1.,1.,2.]),ops);rows.append({'case':case,'R':R.tolist(),'L':L.tolist(),'live':live,'legacy_ops':ops.n})
   saved.append(rows)
  assert saved[0]==saved[1];out.update(rows=saved[1],ledger=ledger.report())
 else:
  old=json.loads((PRIOR/('guard_'+name+'.json')).read_text());model=tuple(np.array(w) for w in old['model']);FX,F3,Ops,ledger=load(True);ops=Ops();stats={};value=F3.est_F2L2_4(model,ops,stats);expected=old['arms'][0]
  assert expected['tier']=='F2L3' and value==expected['estimate'] and ops.n==expected['legacy_ops'] and stats==expected['stats'],(value,ops.n,stats)
  out.update(value=value,legacy_ops=ops.n,stats=stats,ledger=ledger.report())
 out['complete']=True
except Exception as exc:out['failure']=repr(exc);raise
finally:
 out.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('ledger','rows')}),flush=True)
