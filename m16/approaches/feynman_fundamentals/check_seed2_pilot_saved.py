"""Independent saved certificate, bound and count review; no mass evaluations."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,math,time
import numpy as np
import mpmath as mp
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/e46_e48_review/seed2_reference')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def det(a):
 a=[r[:] for r in a];v=F(1)
 for j in range(len(a)):
  pivot=next((i for i in range(j,len(a)) if a[i][j]),None)
  if pivot is None:return F(0)
  if pivot!=j:a[j],a[pivot]=a[pivot],a[j];v=-v
  v*=a[j][j]
  for i in range(j+1,len(a)):
   t=a[i][j]/a[j][j]
   for k in range(j+1,len(a)):a[i][k]-=t*a[j][k]
 return v

def mf(x):
 f=F(x);return mp.mpf(f.numerator)/f.denominator

def main():
 start=time.process_time();protocol=json.loads((P/'pilot_protocol.json').read_text());data=json.loads((P/'pilot.json').read_text());certs=json.loads((P/'pilot_certificates.json').read_text());capture=json.loads((P/'capture.json').read_text());progress=[json.loads(x) for x in (P/'pilot_progress.jsonl').read_text().splitlines()]
 assert protocol['capture_review']=='c8fff4dd' and protocol['export_gate_clearance']
 for p,h in protocol['pins'].items():assert sha(Path(p))==h,p
 assert data['protocol_sha256']==certs['protocol_sha256']==sha(P/'pilot_protocol.json')
 assert data['certificates_sha256']==sha(P/'pilot_certificates.json') and data['complete'] and certs['complete'] and 'failure' not in data
 inds=[i*(37739-1)//63 for i in range(64)];assert data['indices']==protocol['indices']==inds
 assert [r['index'] for r in data['rows']]==[r['index'] for r in certs['rows']]==inds
 selected={};stream=hashlib.sha256();offset=0
 for r in capture['parents']:
  path=P/'parents'/r['archive'];assert sha(path)==r['sha256']
  with np.load(path,allow_pickle=False) as z:
   assert z.files==['rays'];a=z['rays'];assert a.shape==(r['N'],4,4) and a.dtype==np.float64 and np.isfinite(a).all();stream.update(a.tobytes(order='C'))
   for i in inds:
    if offset<=i<offset+len(a):selected[i]=[[F(float(x)) for x in row] for row in a[i-offset]]
   offset+=len(a)
 assert offset==37739 and stream.hexdigest()==protocol['ordered_ray_sha256'] and len(selected)==64
 expected_progress=[{'stage':'load_attempt'}]
 expected_progress += [dict(stage=stage,index=i) for i in inds for stage in ('certificate_attempt','certificate_complete')]
 expected_progress += [dict(stage=stage,index=i,**({'mode':'returned'} if stage=='mass_complete' else {})) for i in inds for stage in ('mass_attempt','mass_complete')]
 expected_progress += [dict(stage='complete',returned=64)]
 assert progress==expected_progress and len(progress)==258
 target=F(1,10**7*37739);tol=data['allocation']['floating'];assert F(tol)<=target and F(data['allocation']['exact'])==target
 value=1e-7/37739;down=F(value)>target
 assert data['allocation']['nextafter_down']==down and tol==(math.nextafter(value,0) if down else value)
 ledger=Counter();methods=Counter();errors=[];values=[];bounds=[];refbounds=[];orders=[];peak=0;construction=Counter();validation=Counter()
 mp.mp.dps=110
 for row,cr in zip(data['rows'],certs['rows'],strict=True):
  idx=row['index'];A=selected[idx];G=[[sum(A[i][k]*A[j][k] for k in range(4)) for j in range(4)] for i in range(4)];c=cr['certificate'];a,b,d=map(F,(c['a'],c['b'],c['det']));y=list(map(F,c['y']));slack=[sum(G[i][j]*y[j] for j in range(4))-a for i in range(4)]
  assert d==det(A)!=0 and len(y)==4 and min(y)>=0 and sum(y)==1 and 0<a<=b
  assert min(slack)>=0 and all(y[i]*slack[i]==0 for i in range(4)) and slack==list(map(F,c['slack']))
  assert sum(y[i]*(slack[i]+a) for i in range(4))==a and b==max(G[i][i] for i in range(4))
  assert cr['independent_validation']['passed'] is True
  for name,obj in [('construction',cr['construction_inventory']),('validation',cr['independent_validation']['exact_inventory'])]:
   assert obj['big_integer_internal_cost_unmeasured'] and all(type(n) is int and n>=0 for n in obj['operations'].values());(construction if name=='construction' else validation).update(obj['operations']);peak=max(peak,obj['max_numerator_bits'],obj['max_denominator_bits'])
  ref=row['reference'];rb=row['reference_bound'];assert ref['determinant_float']==float(abs(d)) and ref['kernel_arithmetic']==39*32**3+1 and ref['value']>0
  am,bm,dm=mf(a),mf(b),abs(mf(d));s=mp.sqrt(am*bm);rho=(mp.sqrt(bm)-mp.sqrt(am))/(mp.sqrt(bm)+mp.sqrt(am));m=31;alpha=(am+bm)/2
  tail=2*rho**m/s**3*((alpha+s*m)/(1-rho)+s*rho/(1-rho)**2)
  qb=dm*tail/(6*mp.pi**2);db=abs(mf(F(ref['determinant_float']))-dm)/(12*mp.pi**2*am**2);total=qb+db
  for key,exact in [('quadrature_bound',qb),('determinant_factor_bound',db),('total_bound',total)]:
   assert abs(mf(rb[key])-exact)<=abs(exact)*mp.mpf('3e-15')+mp.mpf('1e-320'),(idx,key)
  assert abs(mp.mpf(rb['total_decimal'])-total)<abs(total)*mp.mpf('1e-75')+mp.mpf('1e-320')
  assert rb['order']==32 and rb['qualifies']==bool(total<=mf(target)) and rb['ordinary_mp_conditional']
  cand=row['candidate'];center=cand['center'];assert cand['mode']=='returned' and cand['bound']<=tol and row['bound_pass']
  methods[cand['method']]+=1;trials=center['trials']
  if cand['method']=='center_interval':
   assert center['mode']=='interval' and not trials and cand['kernel_counts']=={}
   assert cand['value']==sum(center['interval'])/2 and cand['bound']==(center['interval'][1]-center['interval'][0])/2
   counts={'arithmetic':181,'comparison':11,'sqrt':0,'abs':1}
  else:
   assert cand['method']=='center_quadrature' and center['mode']=='quadrature'
   ns=[x['order'] for x in trials];assert ns==[n for n in (2,3,4,6,8,12,16,20,24,32) if n<=cand['order']]
   assert trials[-1]['order']==center['order']==cand['order'] and trials[-1]['bound']==center['bound']
   assert all(t['bound']>tol for t in trials[:-1]) and trials[-1]['bound']<=tol
   counts={'arithmetic':189+sum(n+11 for n in ns),'comparison':11+len(ns),'sqrt':3,'abs':1}
   assert cand['bound']==center['bound']+cand['series_error_bound']
   n=cand['order'];nn=n*n;k=cand['kernel_counts'];large=k.get('sin',0);assert 0<=large<=nn and cand['inner_modes']=={'analytic':nn}
   kexpected={'arithmetic':132*nn+5-16*large,'comparison':37*nn-5+(3*nn if large else 0),'sqrt':4*nn,'atan2':nn,'abs':1}
   if large:kexpected['sin']=large
   assert k==kexpected,(idx,k,kexpected);assert cand['analytic_float_error_certified'] is False
   if n not in orders:orders.append(n)
  assert center['setup_selection_counts']==counts,(idx,center['setup_selection_counts'],counts)
  ledger.update(counts);ledger.update(cand['kernel_counts'])
  diff=cand['value']-ref['value'];assert row['difference']==diff and row['comparison_pass']==(abs(diff)<=cand['bound']+math.nextafter(rb['total_bound'],math.inf)+5e-14) is True
  errors.append(diff);values.append(cand['value']);bounds.append(cand['bound']);refbounds.append(rb['total_bound'])
 assert [r['order'] for r in data['candidate_rule_events']]==orders
 rulecount=sum(3*n+7*n*n for n in orders)
 for r in data['candidate_rule_events']:assert r['counts']=={'arithmetic':3*r['order']+7*r['order']**2} and r['legendre_internal_cost_unknown']
 assert data['candidate_rule_counts']=={'arithmetic':rulecount};ledger['arithmetic']+=rulecount;ledger['universal_constant_arithmetic']+=6
 assert dict(ledger)==data['known_candidate_counts'] and sum(ledger.values())==data['known_candidate_total']==246787
 assert data['reference_rule_arithmetic']==18*32**3+3*32==589920 and data['reference_kernel_arithmetic']==64*(39*32**3+1)==81788992
 assert len(data['reference_rule_events'])==1 and data['reference_universal_constant_arithmetic']==3
 assert data['methods']==dict(methods)=={'center_quadrature':36,'center_interval':28} and data['returned']==64 and data['unresolved_indices']==[]
 for key,v in [('covered_candidate_probability',math.fsum(values)),('covered_candidate_bound',math.fsum(bounds)),('difference_l1',math.fsum(map(abs,errors))),('maximum_absolute_difference',max(map(abs,errors)))]:assert data[key]==v,key
 assert data['reference_loose_bounds']==sum(x>tol for x in refbounds)==0 and data['full_floating_certificate'] is False and data['full_cost'] is False
 result={'complete':True,'owner_artifact_commit':'223d2c90','raw_sha256':sha(P/'pilot.json'),'certificates_sha256':sha(P/'pilot_certificates.json'),'protocol_sha256':sha(P/'pilot_protocol.json'),'checker_sha256':sha(Path(__file__)),'exact_kkt_and_independent_elimination_determinants':64,'saved_progress_records':258,'methods':dict(methods),'known_candidate_counts':dict(ledger),'known_candidate_total':sum(ledger.values()),'reference_kernel_arithmetic':81788992,'reference_rule_arithmetic':589920,'difference_l1':math.fsum(map(abs,errors)),'maximum_absolute_difference':max(map(abs,errors)),'candidate_bound_sum':math.fsum(bounds),'maximum_reference_bound':max(refbounds),'recorded_construction_operations':dict(construction),'recorded_validation_operations':dict(validation),'recorded_peak_rational_bits':peak,'author_cpu':data['cpu'],'review_cpu':time.process_time()-start,'mass_model_geometry_candidate_calls':0,'qualification':['64 fixed selected rays only; no full-model accuracy or sampling comparison.','KKT range and determinant certified for saved dyadic rays; saved Gaussian integrations not replayed.','Reference tail recomputed independently at110digits, remains ordinary-MP conditional, not directed interval.','Source kernel formulas reconstructed conditional on recorded all-analytic/sin branch census; full hardware costs and actual floating evaluation error excluded.','Exact construction/validation operation inventories conserved, not replayed per operation; big-integer internals unpriced.']}
 (HERE/'seed2_pilot_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['complete','exact_kkt_and_independent_elimination_determinants','known_candidate_total','difference_l1','review_cpu']}))
if __name__=='__main__':main()
