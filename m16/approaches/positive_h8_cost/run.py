"""Frozen64 H8T4 pilot with unchanged captured geometry and prior references."""
import time
START=time.process_time()
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib,importlib.util,json,math,resource,sys
import numpy as np
P=Path(__file__).resolve().parent
PROTOCOL=json.loads((P/'protocol.json').read_text())
OLD=P.parent/'positive_mass_budget/primitive.py'
KERNEL=P.parent/'analytic_inner_mass/primitive.py'
RESCUE=Path('/workspace/work/recover-small_h2_56/m16/approaches/radial_bound_rescue')
REF=Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review')
TOL=1e-7/35098
PI12=12*math.pi**2;PI6=6*math.pi**2;PI2=2*math.pi**2

def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def load():
 for p,h in PROTOCOL['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 old=module('h8_positive',OLD);kernel=module('h8_inner',KERNEL)
 sys.path.insert(0,str(RESCUE))
 try:rescue=module('h8_exact_rescue',RESCUE/'run.py')
 finally:sys.path.pop(0)
 radial,_,_=rescue.load();rescue.TOLERANCE=Fraction(1,10**7*35098)
 return old,kernel,rescue,radial

def select(R,old):
 op=old.Counter();out=dict(mode='unresolved',trials=[])
 try:
  a,b,det=old.geometry(R,op);out.update(a=a.v,b=b.v,det=det.v)
  scale=det/PI12;lo=scale/(b*b);hi=scale/(a*a);mid=(lo+hi)/2;err=(hi-lo)/2
  out['interval']=[lo.v,hi.v]
  if err<=TOL:out.update(mode='interval',value=mid.v,bound=err.v)
  else:
   alpha=(a+b)/2;delta=(a*b).root();sa=a.root();sb=b.root();rho=(sb-sa)/(sb+sa);one=1-rho;scale=det/PI6
   for n in PROTOCOL['orders']:
    power=rho
    for _ in range(n-2):power=power*rho
    tail=2*power/(delta*delta*delta)*(alpha/one+delta*((n-1)/one+rho/(one*one)));bd=scale*tail
    out['trials'].append(dict(order=n,bound=bd.v))
    if bd<=TOL:out.update(mode='quadrature',order=n,bound=bd.v);break
   else:out['reason']='Center bound misses finite-order target'
 except Exception as exc:out['reason']=type(exc).__name__+': '+str(exc)
 out['setup_selection_counts']=op.dict();return out

def one(R,old,kernel,rescue,radial,rules):
 start=time.process_time();center=select(R,old);out=dict(center=center)
 if center['mode']=='interval':out.update(mode='returned',method='center_interval',value=center['value'],bound=center['bound'],kernel_counts={})
 elif center['mode']=='quadrature':
  try:
   result=kernel.integrate(R,center['order'],center['det'],rules);out.update(result,mode='returned',method='center_quadrature',bound=center['bound']+result['series_error_bound'])
   if out['bound']>TOL:out.update(mode='unresolved',reason='Series remainder exceeds target')
  except Exception as exc:out.update(mode='failure',reason=type(exc).__name__+': '+str(exc))
 else:
  det=center.get('det')
  if det is None:
   op=old.Counter();A=[[old.C(x,op) for x in row] for row in R];det=abs(old.determinant(A)).v;out['repeated_determinant_counts']=op.dict()
  try:
   r=rescue.one(R,det,radial,kernel,rules);out['rescue']=r
   if r['mode']=='returned':out.update(mode='returned',method='exact_rescue',value=r['value'],bound=r['conditional_bound'],kernel_counts=r['kernel_counts'])
   else:out.update(mode='unresolved',reason=r['reason'])
  except Exception as exc:out.update(mode='failure',reason=type(exc).__name__+': '+str(exc))
 out['cpu']=time.process_time()-start;return out

def save(name,out,rules):
 count=Counter()
 for r in out['rows']:
  count.update(r['center']['setup_selection_counts']);count.update(r.get('repeated_determinant_counts',{}));count.update(r.get('kernel_counts',{}))
 count.update(rules.ledger.dict());count['universal_constant_arithmetic']+=6
 out.update(cpu_including_imports=time.process_time()-START,known_float_named_counts=dict(count),known_float_named_total=sum(count.values()),cold_rule_counts=rules.ledger.dict(),cold_rules=rules.events,exact_setup_in_known_float_total=False,full_cost=False,full_floating_certificate=False,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (P/name).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

def controls():
 old,kernel,rescue,radial=load();rules=kernel.Rules();rows=[]
 cases=[('orthant',np.eye(4),1/16),('diagonal',np.diag([1.,2.,3.,4.]),1/16),('nonacute',np.array([[1.,0,0,0],[-2,1,0,0],[0,0,1,0],[0,0,0,1]]),math.atan2(1,-2)/(8*math.pi)),('boundary',np.array([[1.,0,0,0],[2,1,0,0],[2,0,1,0],[2,0,0,1]]),None)]
 out=dict(phase='controls',rows=rows,complete=False,per_cone_target=TOL)
 for name,R,truth in cases:
  for power in (-5,0,5):
   if time.process_time()-START>18:out['stop_reason']='18CPU cap';save('controls.json',out,rules);return
   row=one(R*2.**power,old,kernel,rescue,radial,rules);row.update(name=name,power=power,truth=truth)
   if row['mode']=='returned' and truth is not None:
    row['error']=row['value']-truth;row['analytic_pass']=abs(row['error'])<=row['bound']+5e-14
   if row['mode']=='unresolved':
    # Separate validation of a correctly rejected finite-order candidate.
    r=row['rescue'];last=r['stages'][-1]['trials'][-1];check=kernel.integrate(R*2.**power,32,r['old_determinant'],rules)
    row['rejected_candidate_validation']=dict(value=check['value'],bound=last['total_bound']+check['series_error_bound'],kernel_counts=check['kernel_counts'],not_returned_by_estimator=True)
    if truth is not None:
     row['rejected_candidate_validation']['error']=check['value']-truth
     row['rejected_candidate_validation']['passes']=abs(check['value']-truth)<=last['total_bound']+check['series_error_bound']+5e-14
     assert row['rejected_candidate_validation']['passes']
   rows.append(row);save('controls.json',out,rules)
   assert row['mode'] in ('returned','unresolved') and row.get('analytic_pass',True),row
   if row['mode']=='returned':assert row['bound']<=TOL
 # Scale invariance tests validate the new selector across synthetic paths.
 for name,_,_ in cases:
  group=[r for r in rows if r['name']==name];assert len({r['mode'] for r in group})==1
  v=[r['value'] if r['mode']=='returned' else r['rejected_candidate_validation']['value'] for r in group];assert max(v)-min(v)<1e-13
 out.update(complete=True,returned=sum(r['mode']=='returned' for r in rows),unresolved=sum(r['mode']=='unresolved' for r in rows),validation_kernel_counts=dict(sum((Counter(r.get('rejected_candidate_validation',{}).get('kernel_counts',{})) for r in rows),Counter())));save('controls.json',out,rules);print(json.dumps({k:v for k,v in out.items() if k not in ('rows','cold_rules')}))

def pilot():
 assert json.loads((P/'controls.json').read_text())['complete'];old,kernel,rescue,radial=load();rules=kernel.Rules();rays=np.load(REF/'error_captured.npz')['rays'];assert len(rays)==35098
 reference=np.full(35098,np.nan);bounds=np.full(35098,np.nan);seen=np.zeros(35098,int)
 for i in range(8):
  a=np.load(REF/f'error_batch_{i}.npz');reference[a['indices']]=a['positive'];bounds[a['indices']]=a['bounds'];seen[a['indices']]+=1
 assert np.all(seen==1)
 out=dict(phase='fixed64',indices=PROTOCOL['indices'],per_cone_target=TOL,rows=[],complete=False)
 for index in PROTOCOL['indices']:
  if time.process_time()-START>18:out['stop_reason']='18CPU cap';break
  row=one(rays[index],old,kernel,rescue,radial,rules);row.update(index=index,reference=float(reference[index]),reference_bound=float(bounds[index]))
  if row['mode']=='returned':row.update(error=row['value']-row['reference'],comparison_pass=bool(abs(row['value']-row['reference'])<=row['bound']+row['reference_bound']+5e-14))
  out['rows'].append(row);save('pilot.json',out,rules)
  if row['mode']=='failure' or row.get('comparison_pass') is False:out['stop_reason']='Explicit failure/comparison gate';break
 else:out['complete']=True
 out.update(returned=sum(r['mode']=='returned' for r in out['rows']),methods=dict(Counter(r.get('method',r['mode']) for r in out['rows'])),unresolved_indices=[r['index'] for r in out['rows'] if r['mode']!='returned'])
 save('pilot.json',out,rules);print(json.dumps({k:v for k,v in out.items() if k not in ('rows','cold_rules','indices')}))
if __name__=='__main__':
 resource.setrlimit(resource.RLIMIT_CPU,(20,22))
 controls() if sys.argv[1]=='controls' else pilot()
