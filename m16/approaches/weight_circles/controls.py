"""Frozen analytic circle controls only; no trained model or random input."""
import hashlib,importlib.util,itertools,json,math,time,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
TRACER=HERE.parent/'hopf_native_cost/tracer.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dot4(a,b):return (a[0]*b[0]+a[1]*b[1])+(a[2]*b[2]+a[3]*b[3])
def frame(n1,n2):
 n1=tuple(map(float,n1));n2=tuple(map(float,n2));assert len(n1)==len(n2)==4
 if not all(math.isfinite(v) for v in n1+n2):raise ValueError('Nonfinite input')
 s=math.sqrt(dot4(n1,n1))
 if not math.isfinite(s) or s<=0:raise ValueError('Invalid first norm')
 a=tuple(v/s for v in n1);p=dot4(n2,a);r=tuple(v-p*w for v,w in zip(n2,a));s=math.sqrt(dot4(r,r))
 if not math.isfinite(s) or s<=0:raise ValueError('No nonzero residual for rank2 frame')
 b=tuple(v/s for v in r)
 return np.array(a),np.array(b)
def wedge(n1,n2,a,b):
 coefficients=[(dot4(n,a),dot4(n,b)) for n in (n1,n2)];cuts=[0.,2*math.pi]
 for A,B in coefficients:
  assert A!=0 or B!=0
  psi=math.atan2(B,A)
  cuts.extend([(psi+math.pi/2)%(2*math.pi),(psi-math.pi/2)%(2*math.pi)])
 cuts=sorted(set(cuts));total=0.
 for lo,hi in zip(cuts,cuts[1:]):
  t=(lo+hi)/2
  if all(A*math.cos(t)+B*math.sin(t)>=0 for A,B in coefficients):total+=hi-lo
 return total/(2*math.pi)
def models():
 wi=np.array([1.,-1.,1.,-1.]);wh=np.zeros((4,4));wh[2,:2]=[-1,1];wh[3,:2]=[1,-1];wo=np.zeros((3,4));wo[0,2]=wo[1,3]=1.
 diff=(wi,wh,wo)
 wi=np.array([1.,-1.,0.,0.,1.,-1.,1.,-1.,0.,0.]);wh=np.zeros((10,10))
 wh[2,0]=wh[3,1]=wh[8,4]=wh[9,5]=1.
 wh[4,:2]=[-1,1];wh[5,:2]=[1,-1];wh[6,2:4]=[-1,1];wh[7,2:4]=[1,-1]
 wo=np.zeros((3,10));wo[0,[6,7,8,9]]=-1;wo[1,[4,5,8,9]]=-1;wo[2,[4,5,6,7]]=-1
 median=(wi,wh,wo);cycle=(wi.copy(),wh.copy(),wo[[1,2,0]].copy())
 zero=(np.zeros(1),np.zeros((1,1)),np.zeros((4,1)))
 return {'difference':diff,'median':median,'cycle':cycle,'rank_failure':zero}
def explicit_state(name,x):
 x0,x1,x2=map(int,x);values=[x2,x2-x1] if name=='difference' else [x2,x1,x2-x1,x2-x0,x1-x0]
 return np.array([q for v in values for q in (max(v,0),max(-v,0))],dtype=float)
def main():
 start=time.process_time();dst=HERE/'controls.json';assert not dst.exists();gate=json.loads((HERE/'controls_gate.json').read_text());assert gate['complete']
 for p,d in gate['pins'].items():assert sha(Path(p))==d
 spec=importlib.util.spec_from_file_location('t191_fixed_hopf_tracer',TRACER);T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
 out=dict(complete=False,source_sha256=sha(Path(__file__)),gate_sha256=sha(HERE/'controls_gate.json'),wedges=[],degenerate_frames=[],forward=[],circles=[],new_reference_draws=0,new_trained_calls=0,scope='Finite analytic controls and raw source ledgers only. Circle restriction/projection theorem is a real-arithmetic hypothesis; no generic discovery, complete native cost or trained claim.')
 def save():out['cpu']=time.process_time()-start;tmp=dst.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(dst)
 save()
 try:
  Q=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=float)/2
  assert np.array_equal(Q.T@Q,np.eye(4))
  for orient,R in [('identity',np.eye(4)),('hadamard',Q)]:
   e1,e2=R[:,0],R[:,1]
   for k in range(1,8):
    theta=k*math.pi/8;n1=e1;n2=math.cos(theta)*e1+math.sin(theta)*e2;a,b=frame(n1,n2);value=wedge(n1,n2,a,b);expected=(math.pi-theta)/(2*math.pi)
    row=dict(orientation=orient,k=k,n1=n1.tolist(),n2=n2.tolist(),a=a.tolist(),b=b.tolist(),value=value,expected=expected,error=value-expected,frame_kernel_numeric=39,frame_positive_norm_comparisons=2,validation_control_excluded=True);out['wedges'].append(row);save();assert abs(value-expected)<=5e-14
  for sign in (1.,-1.):
   try:frame([1,0,0,0],[sign,0,0,0])
   except ValueError as e:out['degenerate_frames'].append(dict(sign=sign,rejected=True,error=str(e)));save()
   else:raise AssertionError('Exact collinearity not rejected')
  ms=models();out['models']={name:[a.tolist() for a in m] for name,m in ms.items()};save()
  for name in ('difference','median','cycle'):
   wi,wh,wo=ms[name]
   for x in itertools.permutations(range(-2,3),3):
    h=np.zeros(len(wi))
    for v in x:h=np.maximum(wh@h+wi*v,0.)
    target=sorted(range(3),key=lambda j:x[j])[1]
    expected=(0 if x[2]>x[1] else 1) if name=='difference' else (target if name=='median' else (target-1)%3)
    pred=int(np.argmax(wo@h));same=np.array_equal(h,explicit_state(name,x));row=dict(model=name,x=list(x),state_equal=same,pred=pred,expected=expected,target=target);out['forward'].append(row);save();assert same and pred==expected
  a3=np.array([1.,-1.,0.])/math.sqrt(2);b3=np.array([1.,1.,-2.])/math.sqrt(6)
  a4=np.array([0.,1.,-1.,0.])/math.sqrt(2);b4=np.array([0.,1.,1.,-2.])/math.sqrt(6)
  expected_values={'difference':1/3,'median':1.,'cycle':0.,'rank_failure':.5}
  for name,model in ms.items():
   a,b=(a4,b4) if name=='rank_failure' else (a3,b3)
   for phase in (0.,math.pi/7,math.pi/4,math.pi/2):
    aa=a*math.cos(phase)+b*math.sin(phase);bb=-a*math.sin(phase)+b*math.cos(phase)
    plain,_,_=T.load(False,rescue=True);observed,ledger,startup=T.load(True,rescue=True);op=T.Ops();ot=T.Ops()
    baseline=plain.fibre(model,aa,bb,'uniform',op);traced=observed.fibre(model,aa,bb,'uniform',ot)
    row=dict(model=name,phase=phase,a=aa.tolist(),b=bb.tolist(),baseline=list(baseline),traced=list(traced),baseline_hex=float(baseline[0]).hex(),traced_hex=float(traced[0]).hex(),legacy_baseline=op.n,legacy_traced=ot.n,expected_circle=expected_values[name],gaussian_expected=.25 if name=='rank_failure' else expected_values[name],ledger=ledger.report(),startup=startup)
    out['circles'].append(row);save();assert baseline==traced and row['baseline_hex']==row['traced_hex'] and op.n==ot.n
    assert abs(traced[0]-expected_values[name])<=5e-13
  assert len(out['wedges'])==14 and len(out['degenerate_frames'])==2 and len(out['forward'])==180 and len(out['circles'])==16
  out.update(complete=True,circle_calls=32,deterministic_forward_calls=180,frame_success_calls=14,frame_rejection_calls=2);save();print(json.dumps({k:out[k] for k in ('complete','circle_calls','deterministic_forward_calls','cpu')}))
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
