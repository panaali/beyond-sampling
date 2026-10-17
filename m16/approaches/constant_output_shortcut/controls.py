"""Fixed analytic/tie/negative controls; validation inputs never enter estimator."""
import time
START=time.process_time()
from pathlib import Path
from fractions import Fraction as F
import itertools,json,hashlib,resource,math
import numpy as np
import primitive as A
P=Path(__file__).resolve().parent

def model(slopes=None,T=4):
 return (np.array([1.,-1.]) if slopes is not None else np.zeros(2),np.zeros((2,2)),np.repeat(np.asarray(slopes if slopes is not None else [0]*T,dtype=float)[:,None],2,axis=1))
def evaluate(m,x):
 Wi,Wh,Wo=m;h=np.zeros(len(Wi))
 for v in x:h=np.maximum(0,Wh@h+Wi*v)
 return int(np.argmax(Wo@h))
def digest(m):
 d=hashlib.sha256()
 for a in m:d.update(a.tobytes())
 return d.hexdigest()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));proto=json.loads((P/'protocol.json').read_text())
 for f,h in proto['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 cases=[('alloff',model(), 'certified'),('identical_nonzero',model([1,1,1,1]),'certified'),('identical_zero',model([0,0,0,0]),'certified'),('tie_winner0',model([1,1,0,-1]),'certified'),('tie_winner1',model([-1,2,2,0]),'fallback'),('tie_winner2',model([-1,0,3,3]),'fallback'),('strict_winner3',model([-1,0,1,2]),'fallback')]
 for T in (2,3,10):cases.append((f'alloff_T{T}',model(None,T),'certified'))
 cases += [('unequal_dominance',(np.array([1.,-1.]),np.array([[1.,-1.],[-1.,1.]]),np.array([[2.,3.],[2.,2.],[1.,3.],[-1.,-2.]])),'certified'),('unreachable_negative_margin',(np.array([1.,0.]),np.zeros((2,2)),np.array([[1.,-100.],[0.,100.],[0.,0.],[0.,0.]])),'fallback')]
 tiny=model([1,1,0,-1]);tiny[2][0,0]=np.nextafter(1.,0.);cases.append(('one_ulp_violation',tiny,'fallback'))
 off=model([-1,5,0,0]);off[0][:]=0.;cases.append(('alloff_other_row_dominates',off,'certified'))
 zero=model();zero[0][:]=-0.;cases.append(('signed_zero_input',zero,'certified'))
 rows=[]
 for name,m,status in cases:
  before=digest(m);called=[];sentinel=object()
  def fallback(received):called.append(received is m);return sentinel
  out=A.dispatch(m,fallback);cert=out['shortcut'];assert cert['status']==status and digest(m)==before
  assert out['fallback_called']==(status=='fallback');assert len(called)==int(status=='fallback')
  if called:assert called==[True] and out['result'] is sentinel
  T=m[2].shape[0];inputs=list(itertools.product((-1.,0.,1.),repeat=T)) if T<=4 else [tuple([0.]*T)]+[tuple(s if j==i else 0. for j in range(T)) for i in range(T) for s in (-1.,1.)]
  labels=[evaluate(m,x) for x in inputs]
  if status=='certified':assert set(labels)=={0};assert cert['exact_accuracy']==[1,T];assert F(cert['accuracy'])-F(1,T)<=F(1,2**52)
  rows.append(dict(name=name,status=status,certificate=cert,model_weights=[a.tolist() for a in m],validation_forward_inputs=len(inputs),observed_labels=sorted(set(labels)),returned_original_fallback_object=bool(called),model_unchanged=True,not_an_accuracy_sample=True))
 invalid=[]
 for where in range(3):
  for bad in (float('nan'),float('inf'),-float('inf')):
   m=model();m[where].flat[0]=bad;r=A.recognize(m);assert r['status']=='rejected';invalid.append(dict(array=where,value=repr(bad),result=r))
 for name,m in [('float32',tuple(x.astype(np.float32) for x in model())),('wrong_hidden',(np.zeros(2),np.zeros((3,3)),np.zeros((4,2)))),('T1',model(None,1)),('missing_array',model()[:2])]:
  r=A.recognize(m);assert r['status']=='rejected';invalid.append(dict(name=name,result=r))
 previous=json.loads(Path(next(f for f in proto['pins'] if f.endswith('/controls.json'))).read_text());old=next(r for r in previous['models'] if r['name']=='alloff');assert not old['evaluation']['complete']
 out=dict(complete=True,rows=rows,invalid=invalid,prior_alloff_failure_preserved=dict(N=old['capture']['N'],known_mass_total=old['evaluation']['known_mass_total'],complete=False),estimator_random_inputs=0,validation_inputs=sum(r['validation_forward_inputs'] for r in rows),no_geometry_or_mass_calls=True,cpu=time.process_time()-START,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primitive_sha256=hashlib.sha256((P/'primitive.py').read_bytes()).hexdigest())
 assert not (P/'controls.json').exists();(P/'controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,cases=len(rows),invalid=len(invalid),cpu=out['cpu'],alloff=rows[0]['certificate'])))
if __name__=='__main__':main()
