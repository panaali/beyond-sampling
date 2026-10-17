"""Fixed Wh=0 sign/rank controls with rational truth; no sampled inputs."""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;inv=json.loads((HERE/'inventory.json').read_text());SRC=Path(inv['source_root'])
for x in inv['files']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
# Additional original dependencies are pinned in the pre-run protocol artifact.
pins=json.loads((HERE/'analytic_pins.json').read_text())
for path,sha in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
sys.path.insert(0,str(SRC));import fibre4 as FX;import fibre4b as F3
from fiber3 import Ops
assert F3.LOOK==3 and FX.ADAPT is True and FX.NQ_CORR==3 and FX.EMAX==.25
name=sys.argv[1];assert name in ('zero','constant0','plus3_minus0','plus0_minus3')
Wi=np.array([1.,-1.]);Wh=np.zeros((2,2));Wo=np.zeros((4,2))
if name=='zero':Wi[:]=0;expected=.25
elif name=='constant0':Wo[0]=[1.,1.];expected=.25
elif name=='plus3_minus0':Wo[3]=[1.,0.];Wo[0]=[0.,1.];expected=5/16
else:Wo[0]=[1.,0.];Wo[3]=[0.,1.];expected=3/16
model=(Wi,Wh,Wo);start=time.process_time();out=dict(case=name,model=[x.tolist() for x in model],rational_truth={.25:'1/4',5/16:'5/16',3/16:'3/16'}[expected],expected=expected,diagnostic_abs_tolerance=1e-7,arms=[],complete=False,pins=pins)
path=HERE/('analytic_'+name+'.json');path.write_text(json.dumps(out,indent=2)+'\n')
try:
 for tier in ('F2L3','FX'):
  ops=Ops();stats={};t=time.process_time()
  got=F3.est_F2L2_4(model,ops,stats) if tier=='F2L3' else FX.est_fibre4(model,'FX',ops,stats)
  out['arms'].append(dict(tier=tier,estimate=float(got),error=float(got-expected),passes_frozen_tolerance=abs(got-expected)<=1e-7,legacy_ops=ops.n,stats=stats,cpu_seconds=time.process_time()-t))
  path.write_text(json.dumps(out,indent=2)+'\n')
 out.update(complete=True,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,tier_difference=out['arms'][0]['estimate']-out['arms'][1]['estimate'])
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
except Exception as exc:
 out.update(failure=repr(exc),cpu_seconds=time.process_time()-start);path.write_text(json.dumps(out,indent=2)+'\n');raise
