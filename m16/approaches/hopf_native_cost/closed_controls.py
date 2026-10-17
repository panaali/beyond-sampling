"""Recheck unchanged frozen circle/grid arms with source closure attached."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from tracer import ROOT,Ops
from closed_tracer import load,HERE

def main():
 dst=HERE/'closed_controls.json';assert not dst.exists();start=time.process_time();out={'status':'running','rows':[]};save=lambda:dst.write_text(json.dumps(out,indent=2)+'\n');save()
 try:
  old=json.loads((HERE/'controls.json').read_text());assert old['status']=='complete'
  args=json.loads((ROOT/'full_circle_control.json').read_text());model=tuple(np.array(args[k]) for k in ('wi','wh','wo'));a=np.array(args['a']);b=np.array(args['b'])
  models=[model,(np.zeros(2),np.zeros((2,2)),np.zeros((4,2))),(np.ones(2),np.zeros((2,2)),np.array([[1.,1.],[0.,0.],[0.,0.],[0.,0.]])),(np.array([1.,-1.]),np.array([[.25,-.5],[.75,.125]]),np.array([[.25,-1.],[1.,.5],[-1.,.75],[.5,-.25]]))]
  for rescue in (False,True):
   for idx,m in enumerate(models):
    mod,ledger,startup=load(rescue);ops=Ops();result=mod.fibre(m,a,b,ops=ops);prior=next(r for r in old['rows'] if r['case']=='circle' and r['rescue']==rescue and r['model']==idx)
    assert list(result)==prior['value'] and ops.n==prior['legacy'];rep=ledger.report();assert rep['sites']==prior['ledger']['sites']
    out['rows'].append({'case':'circle','model':idx,'rescue':rescue,'value':list(result),'legacy':ops.n,'ledger':rep});save()
   mod,ledger,startup=load(rescue);ops=Ops();result=mod.est_hopf(models[-1],2,ops,nd=3);prior=next(r for r in old['rows'] if r['case']=='base_grid' and r['rescue']==rescue)
   assert result==prior['value'] and ops.n==prior['legacy'];rep=ledger.report();assert rep['sites']==prior['ledger']['sites']
   out['rows'].append({'case':'base_grid','rescue':rescue,'value':result,'legacy':ops.n,'ledger':rep});save()
  out.update(status='complete',cpu_seconds=time.process_time()-start,source_sha256=hashlib.sha256((HERE/'closed_tracer.py').read_bytes()).hexdigest());save();print(json.dumps({'status':out['status'],'rows':len(out['rows']),'cpu_seconds':out['cpu_seconds']}))
 except Exception as e:out.update(status='failed',failure=repr(e),cpu_seconds=time.process_time()-start);save();raise
if __name__=='__main__':main()
