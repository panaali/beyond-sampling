"""Fixed H5 synthetic shape/parity gate; original source, no trained evaluation."""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20))
 dst=HERE/'controls.json'; assert not dst.exists()
 out={'complete':False,'rows':[],'source_sha256':sha(__file__)}
 def save(): dst.write_text(json.dumps(out,indent=2)+'\n')
 save();start=time.process_time()
 try:
  scope=json.loads((HERE/'controls_scope.json').read_text())
  for p,h in scope['pins'].items(): assert sha(p)==h,p
  protocol=json.loads((HERE/'protocol.json').read_text())
  for p,h in {**protocol['source_pins'],**protocol['frozen_pins']}.items(): assert sha(p)==h,p
  root=Path(protocol['tracer_root']);sys.path.insert(0,str(root))
  import tracer,closed_tracer
  assert Path(tracer.__file__).resolve()==root/'tracer.py'
  assert Path(closed_tracer.__file__).resolve()==root/'closed_tracer.py'
  args=json.loads(Path(scope['witness']).read_text());a=np.array(args['a']);b=np.array(args['b'])
  def pad(m):
   wi,wh,wo=m; h=len(wi);return (np.pad(wi,(0,5-h)),np.pad(wh,((0,5-h),(0,5-h))),np.pad(wo,((0,0),(0,5-h))))
  small=[tuple(np.array(args[k],dtype=float) for k in ('wi','wh','wo')),
   (np.zeros(2),np.zeros((2,2)),np.zeros((4,2))),
   (np.ones(2),np.zeros((2,2)),np.array([[1.,1.],[0.,0.],[0.,0.],[0.,0.]])),
   (np.array([1.,-1.]),np.array([[.25,-.5],[.75,.125]]),np.array([[.25,-1.],[1.,.5],[-1.,.75],[.5,-.25]]))]
  dense=(np.array([1.,-.5,.25,.75,-1.]),(np.arange(25).reshape(5,5)%7-3)/8.,(np.arange(20).reshape(4,5)%9-4)/8.)
  models=[pad(m) for m in small]+[dense]
  for idx,m in enumerate(models):
   assert [x.shape for x in m]==[(5,),(5,5),(4,5)]
   row={'case':'circle','model':idx,'status':'started','weights':[x.tolist() for x in m]};out['rows'].append(row);save()
   base,_,_=tracer.load(False,False);bo=tracer.Ops();bv=base.fibre(m,a,b,ops=bo)
   mod,ledger,startup=closed_tracer.load(False);ops=tracer.Ops();v=mod.fibre(m,a,b,ops=ops)
   row.update(baseline=[float(x).hex() for x in bv],traced=[float(x).hex() for x in v],baseline_legacy=bo.n,legacy=ops.n,ledger=ledger.report(),startup=startup)
   assert row['baseline']==row['traced'] and bo.n==ops.n
   if idx==0:row.update(analytic_fraction=args['analytic_fraction'],original_fraction=v[0],difference=v[0]-args['analytic_fraction'],scope='Zero padding preserves true H2 network; original mask defect may depend on representation.')
   row['status']='complete';save()
  row={'case':'dense_grid','nu':2,'nd':3,'status':'started'};out['rows'].append(row);save()
  base,_,_=tracer.load(False,False);bo=tracer.Ops();bv=base.est_hopf(dense,2,bo,nd=3)
  mod,ledger,startup=closed_tracer.load(False);ops=tracer.Ops();v=mod.est_hopf(dense,2,ops,nd=3)
  row.update(baseline=float(bv).hex(),traced=float(v).hex(),baseline_legacy=bo.n,legacy=ops.n,ledger=ledger.report(),startup=startup)
  assert row['baseline']==row['traced'] and bo.n==ops.n
  row['status']='complete';save()
  for outputs in (5,4):
   row={'case':'einsum_shape','outputs':outputs,'status':'started'};out['rows'].append(row);save()
   A=(np.arange(outputs*5).reshape(outputs,5)-3)/8.;B=(np.arange(30).reshape(2,5,3)-11)/16.
   ledger=closed_tracer.Ledger();v=ledger.call('fixed_einsum','np.einsum',np.einsum,'ij,njk->nik',A,B)
   expected=np.array([[[sum(A[i,k]*B[n,k,j] for k in range(5)) for j in range(3)] for i in range(outputs)] for n in range(2)])
   assert np.array_equal(v,expected)
   report=ledger.report();assert report['inclusive_explicit_total']==2*outputs*3*(5+4)
   row.update(status='complete',ledger=report,value=v.tolist(),expected_multiplications=2*outputs*3*5,expected_additions=2*outputs*3*4);save()
  out['complete']=True
 except BaseException as e:out['failure']=repr(e);raise
 finally:
  out.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
 print(json.dumps({'complete':out['complete'],'rows':len(out['rows']),'cpu':out['cpu_seconds']}))
if __name__=='__main__':main()
