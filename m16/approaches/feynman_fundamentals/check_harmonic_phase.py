import sys,time,json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre
ROOT=Path('/workspace/work/recover-small_h4_4/m16/approaches/anisotropic_harmonic')
sys.path.insert(0,str(ROOT))
from space import Space
start=time.process_time();sp=Space(4,2);gx,gw=np.polynomial.legendre.leggauss(32);rows=[];evaluations=0
for sign in (1,-1):
 v=sign*np.array([.3,.4,.5]);v/=np.linalg.norm(v);r=np.hypot(v[1],v[2]);delta=np.arctan2(v[2],v[1]);actual=sp.matrix(v)
 for n,p,mi,mj in ((1,3,-1,2),(2,4,1,-1)):
  def integrand(z):
   global evaluations
   evaluations+=32
   threshold=-v[0]*z/(r*np.sqrt(1-z*z))
   alpha=np.arccos(np.clip(threshold,-1,1))
   angles=delta+alpha*gx
   angular=alpha/(2*np.pi)*np.sum(gw*np.exp(1j*(mj-mi)*angles))
   radial=np.sqrt((2*n+1)*(2*p+1))*eval_legendre(n,2*z-1)*eval_legendre(p,2*z-1)
   return radial*angular
  real=quad(lambda z:integrand(z).real,0,1,points=[r],epsabs=2e-12,epsrel=2e-12)[0]
  imag=quad(lambda z:integrand(z).imag,0,1,points=[r],epsabs=2e-12,epsrel=2e-12)[0]
  reference=complex(real,imag);value=actual[n*5+mi+2,p*5+mj+2]
  row=dict(sign=sign,n=n,p=p,mi=mi,mj=mj,reference=[real,imag],actual=[value.real,value.imag],difference=abs(value-reference));rows.append(row)
  assert row['difference']<1e-10,row
result=dict(rows=rows,cpu_seconds=time.process_time()-start,direct_arc_evaluations=evaluations,trained_model_evaluations=0)
Path(__file__).with_name('harmonic_phase_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
