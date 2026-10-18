"""Independent saved order-selection, directed formulas and complete decision ledger."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,time
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_910/review/chebyshev5_majorant/selection');start=time.process_time();mp.mp.dps=180
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def m(x):
 x=F(x);return mp.mpf(x.numerator)/x.denominator
r=read(P/'results.json');p=read(P/'protocol.json');prior=read(P.parent/'results.json');controls=read(P/'controls.json')
assert r['complete'] and r['active_bound'] is None and r['bounds_sha256']==sha(P/'bounds.jsonl')
for path,h in p['pins'].items():assert sha(path)==h
assert r['source_sha256']==sha(P/'run.py') and r['selector_sha256']==sha(P/'selector.py') and r['controls_sha256']==sha(P/'controls.json')
assert controls['complete'] and len(controls['rows'])==8 and all(z['status']=='passed' for z in controls['rows'])
rows=[json.loads(x) for x in (P/'bounds.jsonl').read_text().splitlines()];cached={tuple([q['panel'],q['fixture'],q['side'],q['order']]):q for q in prior['rows']}
seen=set();origins=Counter();allcounts=Counter();newcounts=Counter()
for i,q in enumerate(rows):
 key=tuple(q['key'])+(q['order'],);assert q['index']==i and key not in seen and q['status']=='completed';seen.add(key);origins[q['origin']]+=1;allcounts.update(q['counts'])
 if q['origin']=='cached':
  assert key in cached and q['upper_bound']==cached[key]['upper_bound'] and q['counts']==cached[key]['counts'] and q['prior_record']==list(key)
 else:
  assert q['origin']=='new' and key not in cached;newcounts.update(q['counts']);a,b=F(q['a_expanded']),F(q['b_expanded']);cert=q['certificate'];assert 0<a<=F(cert['a'])<=F(cert['b'])<=b
  ra,rb=mp.sqrt(m(a)),mp.sqrt(m(b));ss=ra*rb;B=(m(a)+m(b))/(2*ss);rho=(rb-ra)/(rb+ra);k=q['order']-1;assert q['degree']==q['order']-2
  T0=rho**k/(1-rho);T1=k*T0+rho**(k+1)/(1-rho)**2;T2=k*k*T0+2*k*rho**(k+1)/(1-rho)**2+rho**(k+1)*(1+rho)/(1-rho)**3
  tail=(T2+3*B*T1+(3*B*B-1)*T0)/ss**3;scale=(ra+rb)/2;value=abs(m(cert['det']))*tail*scale/(32*mp.pi**2)
  for field,v in [('integer3_tail',tail),('sqrtK_interval',scale),('majorant_tail',scale*tail)]:assert m(q[field][0])<=v<=m(q[field][1])
  assert m(q['lower_formula_endpoint'])<=value<=m(q['upper_bound'])
assert len(rows)==r['evaluations']==400 and origins==Counter(cached=84,new=316)
assert dict(allcounts)==r['fresh_production_bound_counts'] and dict(newcounts)==r['new_research_bound_counts']
used=[];statuses=Counter();selections=[]
for g in r['groups']:
 assert g['status']=='completed';s=g['selection'];statuses[g['panel']+':'+s['status']]+=1
 assert [z['order'] for z in s['steps']]==p['orders'][:len(s['steps'])]
 for ix,step in enumerate(s['steps']):
  total=F();keys=[]
  for e in step['entries']:
   q=rows[e['index']];used.append(e['index']);assert q['key'][:2]==[g['panel'],g['fixture']] and q['order']==step['order'] and e['upper_bound']==q['upper_bound'] and e['origin']==q['origin'];total+=F(e['upper_bound']);keys.append(q['key'][2])
  assert keys==([None] if g['panel']=='original' else [0,1]);assert F(step['bound_sum'])==total and step['qualifies']==(total<=F(p['tolerance'])) and step['status']=='completed'
  if ix<len(s['steps'])-1:assert not step['qualifies']
 if s['status']=='certified_ideal':
  n=s['selected_order'];nchildren=1 if g['panel']=='original' else 2;assert s['steps'][-1]['qualifies'] and n==s['steps'][-1]['order'] and s['bound_sum']==s['steps'][-1]['bound_sum']
  assert s['predicted_nodes']==nchildren*n**4 and s['predicted_source_kernel_proxy']==nchildren*(125+40*n**4) and s['assembly_additions']==nchildren-1
  selections.append(dict(panel=g['panel'],fixture=g['fixture'],order=n,nodes=s['predicted_nodes'],source_proxy=s['predicted_source_kernel_proxy']))
 else:assert s['status']=='capped' and s['maximum_order']==64 and len(s['steps'])==30 and not s['steps'][-1]['qualifies']
assert sorted(used)==list(range(400)) and statuses==Counter({'original:certified_ideal':6,'original:capped':5,'child:certified_ideal':11})
assert r['required_prior_certificate_advice']==prior['reused_certificate_setup']
out=dict(complete=True,groups=22,decisions=400,cached=84,new=316,statuses=dict(statuses),selections=selections,maximum_child_order=max(z['order'] for z in selections if z['panel']=='child'),results_sha256=sha(P/'results.json'),bounds_sha256=sha(P/'bounds.jsonl'),review_cpu=time.process_time()-start,new_mass_calls=0,scope='Every bound/failed prior order, cached reuse, directed formula, exact group sum and predicted cost checked. All11 split groups certify the ideal target byorder30; five original wide groups remain capped64. No quadrature was evaluated, so selected orders are certificate predictions only; sourceFP/native/advice exclusions persist.')
Path(__file__).with_name('chebyshev5_selection_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='selections'}))
