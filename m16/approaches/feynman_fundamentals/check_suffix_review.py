"""Independent all-pair angular splitting and direct Rayleigh integration."""
import sys,time,json,math,hashlib,itertools
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
ROOT=Path('/workspace/work/recover-small_h2_56/m16/approaches/h2_boundary_mechanism')
sys.path.insert(0,str(ROOT))
from inventory import load
from suffix import calculate
OUT=Path(__file__).parent
start=time.process_time(); evaluations=0

def independent(model):
 global evaluations
 wi,wh,wo=model;t=len(wo);m=t-2
 cuts={0.,math.pi/4,math.pi/2,math.pi,5*math.pi/4,3*math.pi/2,2*math.pi}
 for sign in (-1,1):
  matrix=np.column_stack((wh@(wi*(sign*wi>0)),wi))
  normals=list(matrix)
  for mask in itertools.product((0,1),repeat=2):
   mapped=np.diag(mask)@matrix
   normals.extend((wo[j]-wo[k])@mapped for j in range(t) for k in range(j))
  for v in normals:
   if np.any(v):
    angle=math.atan2(-v[0],v[1])%(2*math.pi)
    for a in (angle,(angle+math.pi)%(2*math.pi)):
     if sign*math.cos(a)>=0:cuts.add(a)
 cuts=sorted(cuts);gx,gw=np.polynomial.legendre.leggauss(24);value=0.;maxerr=0.;nodes=0
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  theta=(lo+hi)/2;v=np.array([math.cos(theta),math.sin(theta)])
  h=np.maximum(wi*v[0],0);h=np.maximum(wh@h+wi*v[1],0);label=int(np.argmax(wo@h))
  for x,w in zip(gx,gw):
   a=(lo+hi)/2+(hi-lo)*x/2;c,s=math.cos(a),math.sin(a);upper=max(c,s);lower=min(c,s)
   def integrand(r):
    global evaluations
    evaluations+=1
    qu=float(ndtr(r*upper));ql=float(ndtr(r*lower))
    if label<t-2:truth=1/m-qu**(m-1)+qu**m-ql**m/m
    elif (label==t-2 and c>=s) or (label==t-1 and s>=c):truth=m*(1-qu)*qu**(m-1)
    else:truth=ql**m
    return r*math.exp(-r*r/2)*truth
   p,e=quad(integrand,0,12,epsabs=2e-12,epsrel=2e-12)
   value+=(hi-lo)*w*p/(4*math.pi);maxerr=max(maxerr,e);nodes+=1
 return value,dict(sectors=len(cuts)-1,angular_nodes=nodes,largest_reported_radial_error=maxerr)

base,_=load(4);rng=np.random.default_rng(47021);cases=[]
refs={r['name']:r for r in json.loads((ROOT/'transfer.json').read_text())['cases']}
for sigma in (.02,.10):
 for i in range(4):
  model=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base)
  if i==0:cases.append((f'T4_rms{sigma}_copy{i}',model,None))
model,_=load(5,2);cases.append(('T5_seed2',model,None))
cases.append(('constant',(base[0],base[1],np.ones_like(base[2])),.25))
wi=np.ones(2);wh=np.zeros((2,2));wo=np.zeros((4,2));wo[-1]=1
cases.append(('last_input',(wi,wh,wo),.3125))
result={'method':'all pairs/all masks + independent GL24 + direct Rayleigh quad','cases':[], 'source_sha256':hashlib.sha256((ROOT/'suffix.py').read_bytes()).hexdigest()}
for name,model,truth in cases:
 p,stats=independent(model);author,_=calculate(model,16)
 row=dict(name=name,independent=p,author=author,difference=p-author,exact_control=truth,**stats)
 if name in refs:row.update(full_reference=refs[name]['reference'],surrogate_bias=p-refs[name]['reference'],saved_mse_ratio=refs[name]['mse_ratio'])
 if truth is not None:row['control_error']=p-truth
 result['cases'].append(row)
 result.update(cpu_seconds=time.process_time()-start,radial_integrand_evaluations=evaluations,model_inputs_sampled=0)
 (OUT/'suffix_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(name,p,p-author,row.get('surrogate_bias'),flush=True)
 if time.process_time()-start>18:raise RuntimeError('bounded review CPU guard')
