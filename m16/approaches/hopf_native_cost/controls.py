"""Frozen primitive and complete-circle parity, no trained model or random draw."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from tracer import load,Ops,HERE,ROOT

def pack(result):
 if isinstance(result,tuple):return [pack(x) for x in result]
 if isinstance(result,np.ndarray):return {'dtype':str(result.dtype),'shape':list(result.shape),'data':result.tolist()}
 if isinstance(result,np.generic):return result.item()
 return result

def same(a,b):
 assert pack(a)==pack(b),(pack(a),pack(b))

class Fixed:
 def __init__(self,x):self.x=x;self.at=0
 def standard_normal(self,shape):
  assert shape[1]==self.x.shape[1];v=self.x[self.at:self.at+shape[0]].copy();self.at+=shape[0];assert v.shape==shape;return v

def main():
 dst=HERE/'controls.json';assert not dst.exists();start=time.process_time()
 out={'status':'running','source_sha256':hashlib.sha256((HERE/'tracer.py').read_bytes()).hexdigest(),'rows':[]}
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  coefficients=np.array([[1.,0.],[0.,1.],[1.,-1.],[0.,0.]])
  witness=json.loads((ROOT/'full_circle_control.json').read_text())
  model=tuple(np.array(witness[k]) for k in ('wi','wh','wo'));a=np.array(witness['a']);b=np.array(witness['b'])
  for rescue in (False,True):
   for lower in (0.,np.pi):
    base,_,_=load(False,rescue);mod,ledger,startup=load(True,rescue)
    args=(np.full(4,lower),np.full(4,2*np.pi),coefficients,np.zeros_like(coefficients))
    same(base._cuts(*args),mod._cuts(*args))
    out['rows'].append({'case':'roots','rescue':rescue,'lower':lower,'ledger':ledger.report(),'startup':startup});save()
   models=[model,(np.zeros(2),np.zeros((2,2)),np.zeros((4,2))),(np.ones(2),np.zeros((2,2)),np.array([[1.,1.],[0.,0.],[0.,0.],[0.,0.]])),(np.array([1.,-1.]),np.array([[.25,-.5],[.75,.125]]),np.array([[.25,-1.],[1.,.5],[-1.,.75],[.5,-.25]]))]
   for idx,m in enumerate(models):
    base,_,_=load(False,rescue);mod,ledger,startup=load(True,rescue);ob=Ops();oc=Ops()
    result=mod.fibre(m,a,b,ops=oc);expected=base.fibre(m,a,b,ops=ob);same(expected,result);assert oc.n==ob.n
    if idx==0:assert result[0]==witness['tile_rescue_fraction' if rescue else 'original_fraction']
    out['rows'].append({'case':'circle','model':idx,'rescue':rescue,'value':pack(result),'legacy':oc.n,'ledger':ledger.report(),'startup':startup});save()
   # Fixed small Hopf base checks accumulation and actually executed geometry.
   base,_,_=load(False,rescue);mod,ledger,startup=load(True,rescue);ob=Ops();oc=Ops()
   value=mod.est_hopf(models[-1],2,oc,nd=3);expected=base.est_hopf(models[-1],2,ob,nd=3);same(expected,value);assert oc.n==ob.n
   out['rows'].append({'case':'base_grid','rescue':rescue,'value':value,'legacy':oc.n,'ledger':ledger.report(),'startup':startup});save()
  # Exercise actual unchanged sampler on a finite deterministic input list.
  x=np.array([[.1,.4,-.5,.3],[2.,-1.,0.,1.],[-2.,1.,3.,0.],[0.,0.,0.,0.],[1.,1.,1.,1.]])
  m=(np.array([1.,-.25,.75,-1.]),np.eye(4)*.25,np.array([[.1,.3,.4,-.5],[.5,-.2,.3,.1],[-.3,.4,.1,.2],[.2,.7,-.4,.5]]))
  base,_,_=load(False,sampler=True);mod,ledger,startup=load(True,sampler=True)
  value=mod.mc_acc(m,5,Fixed(x),chunk=3);expected=base.mc_acc(m,5,Fixed(x),chunk=3);same(value,expected)
  out['rows'].append({'case':'sampler','value':value,'draws':0,'fixed_inputs':5,'ledger':ledger.report(),'startup':startup});save()
  out.update(status='complete',cpu_seconds=time.process_time()-start);save();print(json.dumps({'status':out['status'],'rows':len(out['rows']),'cpu_seconds':out['cpu_seconds']}))
 except Exception as e:
  out.update(status='failed',failure=repr(e),cpu_seconds=time.process_time()-start);save();raise
if __name__=='__main__':main()
