"""Full saved coverage/cost contraction and independent 158 exact certificates."""
from check_seed2_pilot_saved import P,HERE,sha,det,mf
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,math,time,hashlib
import numpy as np
import mpmath as mp
def load(p):return json.loads(p.read_text())
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=2e-14,abs_tol=1e-320),(a,b)
start=time.process_time();s=load(P/'full_summary.json');proto=load(P/'full_protocol.json');pilot=load(P/'pilot.json');cap=load(P/'capture.json')
assert s['complete'] and s['N']==37739 and s['protocol_sha256']==sha(P/'full_protocol.json')
saved={r['index']:r['candidate'] for r in pilot['rows']};rows=[];events=0
for name,h in s['shard_hashes'].items():
 p=P/'full'/name;assert sha(p)==h
 raw=[json.loads(l) for l in p.read_text().splitlines()];assert len(raw)%2==0
 for attempt,result in zip(raw[::2],raw[1::2],strict=True):
  i=len(rows);assert attempt=={'stage':'attempt','index':i,'protocol_sha256':s['protocol_sha256']}
  assert result['stage']=='result' and result['protocol_sha256']==s['protocol_sha256'];r=result['row'];assert r['index']==i
  assert r['reused_pilot']==(i in saved)
  if i in saved:assert r['candidate']==saved[i]
  rows.append(r)
 events+=len(raw)
assert len(rows)==37739 and len(s['shard_hashes'])==148 and events==75478
sessions=[]
for name,h in s['session_hashes'].items():
 p=P/'full'/name;assert sha(p)==h;sess=load(p);assert sess['status']=='complete' and sess['protocol_sha256']==s['protocol_sha256'];sessions.append(sess)
assert len(sessions)==1 and sessions[0]['rows_added']==37739 and sessions[0]['first_index']==0
assert sessions[0]['process_cpu']<110 and sessions[0]['maxrss_kib']<2_000_000
target=F(1,377390000000);tol=sessions[0]['allocation']['floating'];assert F(tol)<=target
rescue_indices=[r['index'] for r in rows if 'rescue' in r['candidate']];assert len(rescue_indices)==158
selected={};offset=0;stream=hashlib.sha256()
for parent in cap['parents']:
 p=P/'parents'/parent['archive'];assert sha(p)==parent['sha256']
 with np.load(p,allow_pickle=False) as z:
  A=z['rays'];assert A.shape==(parent['N'],4,4) and A.dtype==np.float64;stream.update(A.tobytes())
  for i in rescue_indices:
   if offset<=i<offset+len(A):selected[i]=[[F(float(x)) for x in row] for row in A[i-offset]]
  offset+=len(A)
assert offset==37739 and len(selected)==158 and stream.hexdigest()==proto['ordered_ray_sha256']
counts=Counter();methods=Counter();orders=Counter();exactops=Counter();values=[];bounds=[];certs=[]
mp.mp.dps=110
for r in rows:
 c=r['candidate'];b=c['center'];idx=r['index'];assert c['mode']=='returned' and r['bound_pass'] and 0<=c['bound']<=tol and math.isfinite(c['value'])
 methods[c['method']]+=1;values.append(c['value']);bounds.append(c['bound'])
 if b['mode']=='interval':
  expect={'arithmetic':181,'comparison':11,'sqrt':0,'abs':1};assert c['value']==sum(b['interval'])/2 and c['bound']==(b['interval'][1]-b['interval'][0])/2
 elif not b['trials']:
  assert c['method']=='exact_rescue';expect={'arithmetic':170,'comparison':7,'sqrt':0,'abs':1};assert c['repeated_determinant_counts']=={'arithmetic':95,'comparison':0,'sqrt':0,'abs':1}
 else:
  ns=[x['order'] for x in b['trials']];assert ns==list((2,3,4,6,8,12,16,20,24,32)[:len(ns)])
  expect={'arithmetic':189+sum(n+11 for n in ns),'comparison':11+len(ns),'sqrt':3,'abs':1}
  assert all(x['bound']>tol for x in (b['trials'] if b['mode']=='unresolved' else b['trials'][:-1]))
 assert b['setup_selection_counts']==expect;counts.update(expect);counts.update(c.get('repeated_determinant_counts',{}))
 if c['method']!='center_interval':
  q=c.get('rescue',c);n=q['order'];nn=n*n;k=c['kernel_counts'];large=k.get('sin',0)
  assert q['inner_modes']=={'analytic':nn} and 0<=large<=nn
  expected={'arithmetic':132*nn+5-16*large,'comparison':37*nn-5+(3*nn if large else 0),'sqrt':4*nn,'atan2':nn,'abs':1}
  if large:expected['sin']=large
  assert k==expected,(idx,k,expected);counts.update(k)
  if c.get('order') is not None:orders[c['order']]+=1
 if 'rescue' in c:
  q=c['rescue'];assert [x['method'] for x in q['stages']]==['acute_rejected','exact_kkt']
  A=selected[idx];G=[[sum(A[i][k]*A[j][k] for k in range(4)) for j in range(4)] for i in range(4)];d=det(A);ac=q['stages'][0]['certificate'];z=q['stages'][1]['certificate']
  assert [[F(x) for x in row] for row in ac['gram']]==G and not ac['acute'] and any(x<0 for row in G for x in row)
  a,bm,dd=map(F,(z['a'],z['b'],z['det']));y=list(map(F,z['y']));assert dd==d!=0 and bm==max(G[i][i] for i in range(4)) and a>0
  sl=[sum(G[i][j]*y[j] for j in range(4))-a for i in range(4)]
  assert min(y)>=0 and sum(y)==1 and sl==list(map(F,z['slack'])) and min(sl)>=0 and all(y[i]*sl[i]==0 for i in range(4))
  aa,bb,dm=mf(a),mf(bm),abs(mf(d));delta=mp.sqrt(aa*bb);rho=(mp.sqrt(bb)-mp.sqrt(aa))/(mp.sqrt(bb)+mp.sqrt(aa));alpha=(aa+bb)/2
  trials=q['stages'][1]['trials'];assert [t['order'] for t in trials]==list((2,3,4,6,8,12,16,20,24,32)[:len(trials)])
  for t in trials:
   m=t['order']-1;tail=2*rho**m/delta**3*((alpha+delta*m)/(1-rho)+delta*rho/(1-rho)**2)
   qb=dm*tail/(6*mp.pi**2);db=abs(mf(q['old_determinant'])-dm)/(12*mp.pi**2*aa**2);total=qb+db
   for key,v in [('quadrature_bound',qb),('determinant_factor_bound',db),('total_bound',total)]:close(t[key],v)
   assert abs(mp.mpf(t['total_decimal'])-total)<abs(total)*mp.mpf('2e-75')+mp.mpf('1e-320')
   assert t['qualifies']==bool(total<=mf(target))
  assert not any(t['qualifies'] for t in trials[:-1]) and trials[-1]['qualifies'] and trials[-1]['order']==q['order']
  combined=mp.mpf(trials[-1]['total_decimal'])+mp.mpf(q['series_error_bound'])
  assert c['bound']==q['conditional_bound']==math.nextafter(float(combined),math.inf) and c['value']==q['value'] and c['kernel_counts']==q['kernel_counts']
  exactops.update(q['exact_operation_inventory']['operations']);certs.append({'index':idx,'order':q['order'],'exact_kkt':True,'independent_tail':True})
 elif c['method']=='center_quadrature':assert c['bound']==b['bound']+c['series_error_bound']
rule_events=pilot['candidate_rule_events']+[e for x in sessions for e in x['rule_events']];canonical={};staged=0
for e in rule_events:
 n=e['order'];assert e['counts']=={'arithmetic':3*n+7*n*n};staged+=sum(e['counts'].values());canonical[n]=e['counts']
for e in canonical.values():counts.update(e)
counts['universal_constant_arithmetic']+=6
assert dict(counts)==s['known_float_counts'] and sum(counts.values())==s['known_float_total']==156550352
assert staged==s['staged_rule_total']==22892 and dict(methods)==s['methods'] and {str(k):v for k,v in orders.items()}==s['orders']
assert math.fsum(values)==s['covered_error_probability'] and math.fsum(bounds)==s['conditional_bound_sum'] and 1-math.fsum(values)==s['accuracy']
out={'complete':True,'author_commit':'c6ffbf6a','summary_sha256':sha(P/'full_summary.json'),'protocol_sha256':sha(P/'full_protocol.json'),'coverage':37739,'shards':148,'events':events,'pilot_reused_once':64,'exact_original_ray_kkt_certificates':certs,'exact_operation_inventory_sum':dict(exactops),'methods':dict(methods),'known_float_total':sum(counts.values()),'staged_rule_total':staged,'accuracy':s['accuracy'],'conditional_bound_sum':s['conditional_bound_sum'],'author_process_cpu':sessions[0]['process_cpu'],'review_cpu':time.process_time()-start,'scope':'All raw coverage/reuse/count categories plus158 exact saved-ray KKT/determinants and independent110digit bound contractions. No mass evaluation replay; source floating geometry/evaluation and ordinaryMP bound-rounding errors remain excluded. No matched-cost/rung claim.'}
(HERE/'seed2_full_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['complete','coverage','known_float_total','accuracy','conditional_bound_sum','review_cpu']}))
