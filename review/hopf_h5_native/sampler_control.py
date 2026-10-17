"""Fixed five-input H5 sampler arithmetic check; no random draws."""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Fixed:
 def __init__(self,x):self.x=x;self.at=0
 def standard_normal(self,shape):
  assert shape[1]==4
  v=self.x[self.at:self.at+shape[0]].copy();self.at+=shape[0];assert v.shape==shape;return v

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));dst=HERE/'sampler_control.json';assert not dst.exists()
 out={'complete':False,'source_sha256':sha(__file__),'random_draws':0};start=time.process_time()
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  scope=json.loads((HERE/'sampler_scope.json').read_text())
  for p,h in scope['pins'].items():assert sha(p)==h,p
  protocol=json.loads((HERE/'protocol.json').read_text())
  for p,h in {**protocol['source_pins'],**protocol['frozen_pins']}.items():assert sha(p)==h,p
  root=Path(protocol['tracer_root']);sys.path.insert(0,str(root));import tracer,closed_tracer
  assert Path(tracer.__file__).resolve()==root/'tracer.py' and Path(closed_tracer.__file__).resolve()==root/'closed_tracer.py'
  controls=json.loads((HERE/'controls.json').read_text());row=next(r for r in controls['rows'] if r['case']=='circle' and r['model']==4);model=tuple(np.array(x) for x in row['weights'])
  x=np.array([[.1,.4,-.5,.3],[2.,-1.,0.,1.],[-2.,1.,3.,0.],[0.,0.,0.,0.],[1.,1.,1.,1.]])
  base,_,_=tracer.load(False,sampler=True);mod,_,startup=tracer.load(True,sampler=True);ledger=closed_tracer.Ledger();mod._arithmetic=ledger
  r0,r1=Fixed(x),Fixed(x);bv=base.mc_acc(model,5,r0,chunk=3);v=mod.mc_acc(model,5,r1,chunk=3)
  report=ledger.report();out.update(baseline_hex=float(bv).hex(),traced_hex=float(v).hex(),fixed_inputs=x.tolist(),weights=[a.tolist() for a in model],ledger=report,startup=startup,draws_consumed=[r0.at,r1.at],expected_explicit=281*5+8*2+2,dense_indicator=280,zero_state_optimized_indicator=230)
  save();assert out['baseline_hex']==out['traced_hex'] and r0.at==r1.at==5
  assert report['inclusive_explicit_total']==out['expected_explicit']
  out['complete']=True
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps({'complete':out['complete'],'explicit':out['expected_explicit'],'cpu':out['cpu_seconds']}))
if __name__=='__main__':main()
