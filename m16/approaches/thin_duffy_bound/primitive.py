"""Exact stored-ray radial ranges; fixed GL10 real-arithmetic truncation only."""
from pathlib import Path
import importlib.util,json,hashlib,time
from fractions import Fraction
import mpmath as mp
P=Path(__file__).resolve().parent
PROTOCOL=json.loads((P/'protocol.json').read_text())
def load():
 for p,h in PROTOCOL['source_pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 tracker=module('duffy_exact_tracking',Path(PROTOCOL['tracking_source']));radial=module('duffy_exact_radial',Path(PROTOCOL['radial_source']));radial.F=tracker.Tracked
 return tracker,radial

def certificate(R,tracker,radial,reused=None):
 start=time.process_time();tr=tracker.Tracker();T=tracker.Tracked;T.current=tr
 A=[[T(float(x)) for x in row] for row in R];G=[[sum(x*y for x,y in zip(a,b)) for b in A] for a in A]
 determinant=radial.determinant(A);norms=[G[i][i] for i in range(4)];b=max(norms);row=[sum(g) for g in G];total=sum(row);minimum=min(row)
 out=dict(mode='uncovered',lower_candidates={},det=str(determinant),b=str(b))
 if not determinant:out['reason']='Singular stored rays'
 else:
  candidates={}
  if minimum>0 and total>0:candidates['center']=minimum*minimum/total
  acute=all(G[i][j]>=0 for i in range(4) for j in range(i+1,4));out['acute']=acute
  if acute:candidates['acute']=min(norms)/4
  if reused is not None:
   cert=reused['certificate'];assert determinant==T(cert['det']);assert b==T(cert['b'])
   if reused['method']=='exact_kkt':
    a=T(cert['a']);y=[T(x) for x in cert['y']];assert sum(y)==1 and min(y)>=0
    slack=[sum(G[i][j]*y[j] for j in range(4))-a for i in range(4)]
    assert min(slack)>=0 and all(y[i]*slack[i]==0 for i in range(4))
    candidates['reused_kkt']=a
   elif reused['method']=='acute':
    assert acute and T(cert['a'])==min(norms)/4
   else:raise AssertionError(reused['method'])
   out['reused_method']=reused['method']
  out['lower_candidates']={k:str(v) for k,v in candidates.items()}
  if candidates:
   method=max(candidates,key=candidates.get);a=candidates[method];assert a>0 and a<=b
   out.update(mode='bounded',a=str(a),lower_method=method)
  else:out['reason']='No positive center/acute bound or previously certified KKT available'
 out.update(exact_inventory=tr.report(),certificate_cpu=time.process_time()-start)
 return out

def bound(cert,dps=80):
 start=time.process_time()
 def f(s):v=Fraction(s);return mp.mpf(v.numerator)/v.denominator
 with mp.workdps(dps):
  a,b=f(cert['a']),f(cert['b']);d=abs(f(cert['det']));assert 0<a<=b
  alpha=(a+b)/2;delta=mp.sqrt(a*b);rho=(mp.sqrt(b)-mp.sqrt(a))/(mp.sqrt(b)+mp.sqrt(a));N=8
  tail=2*rho**(N+1)/delta**3*(alpha/(1-rho)+delta*((N+1)/(1-rho)+rho/(1-rho)**2))
  v=d*tail/(6*mp.pi**2)
  out=dict(order=10,polynomial_degree=N,bound=float(v),bound_decimal=mp.nstr(v,dps),rho=mp.nstr(rho,dps),mp_dps=dps,qualifies=bool(v<=mp.mpf(1)/(10**7*35098)),directed_floating_certificate=False)
 out['bound_cpu']=time.process_time()-start;return out
