"""Independent whole-corpus saved ledger, coverage and exact rescue certificate audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from itertools import permutations
import gzip,json,hashlib,time,math,numpy as np,mpmath as mp
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_34/m16/approaches/positive_h8_full');REF=Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review');S=json.loads((P/'summary.json').read_text());pilot=json.loads((P.parent/'positive_h8_cost/pilot.json').read_text());old={r['index']:r for r in pilot['rows']};rays=np.load(REF/'error_captured.npz')['rays'];refs={}
for i in range(8):
 with np.load(REF/f'error_batch_{i}.npz') as d:refs.update(zip(d['indices'].tolist(),d['positive'].tolist()))
rows=[]
for shard in S['shards']:
 p=P/shard['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==shard['sha256'];d=json.loads(gzip.decompress(p.read_bytes()));assert d['complete'];rows.extend(d['rows'])
assert [r['index'] for r in rows]==list(range(35098));known=Counter();exact=Counter();methods=Counter();bits=[0,0];mass=F(0);refmass=F(0);bounds=F(0);l1=F(0);reused=rescues=0;ps=list(permutations(range(4)));mp.mp.dps=80;m=lambda x:mp.mpf(x.numerator)/x.denominator
for r in rows:
 assert r['mode']=='returned' and r['bound']<=1e-7/35098 and r['reference']==refs[r['index']];assert r['comparison_pass'] and abs(r['value']-r['reference'])<=r['bound']+r['reference_bound']+5e-14
 if r.get('reused_pilot'):
  assert {k:v for k,v in r.items() if k!='reused_pilot'}==old[r['index']];reused+=1
 known.update(r['center']['setup_selection_counts']);known.update(r.get('repeated_determinant_counts',{}));known.update(r.get('kernel_counts',{}));methods[r['method']]+=1;mass+=F(r['value']);refmass+=F(r['reference']);bounds+=F(r['bound']);l1+=abs(F(r['value'])-F(r['reference']))
 if 'rescue' not in r:continue
 rescues+=1;z=r['rescue'];inventory=z['exact_operation_inventory'];exact.update(inventory['operations']);bits=[max(bits[0],inventory['max_numerator_bits']),max(bits[1],inventory['max_denominator_bits'])];c=z['stages'][-1]['certificate'];R=[[F(float(v)) for v in row] for row in rays[r['index']]];G=[[sum(a*b for a,b in zip(x,y)) for y in R] for x in R];a,b=F(c['a']),F(c['b']);det=sum(((-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))*math.prod(R[i][p[i]] for i in range(4)) for p in ps),F(0));assert det==F(c['det']) and det and a>0 and b==max(G[i][i] for i in range(4))
 if z['stages'][-1]['method']=='acute':assert min(G[i][j] for i in range(4) for j in range(i+1,4))>=0 and a==min(G[i][i] for i in range(4))/4
 else:
  y=list(map(F,c['y']));assert sum(y)==1 and min(y)>=0;gy=[sum(x*t for x,t in zip(line,y)) for line in G];slack=[v-a for v in gy];assert sum(t*v for t,v in zip(y,gy))==a and min(slack)>=0 and all(v*t==0 for v,t in zip(slack,y)) and slack==list(map(F,c['slack']))
 n=z['order'];alpha=(m(a)+m(b))/2;delta=mp.sqrt(m(a)*m(b));rho=(mp.sqrt(m(b))-mp.sqrt(m(a)))/(mp.sqrt(m(b))+mp.sqrt(m(a)));tail=2*rho**(n-1)/delta**3*(alpha/(1-rho)+delta*((n-1)/(1-rho)+rho/(1-rho)**2));bd=abs(m(det))*tail/(6*mp.pi**2)+abs(m(F(abs(z['old_determinant'])))-abs(m(det)))/(12*mp.pi**2*m(a)**2);assert math.isclose(float(bd),z['stages'][-1]['trials'][-1]['total_bound'],rel_tol=1e-13,abs_tol=1e-300);assert bd+mp.mpf(z['series_error_bound'])<=mp.mpf(1)/(10**7*35098)
rules={};staged=Counter()
for session in [pilot]+[json.loads(p.read_text()) for p in sorted(P.glob('session_*.json'))]:
 for e in session.get('cold_rule_events',session.get('cold_rules',[])):
  if e['order'] in rules:assert rules[e['order']]==e['counts']
  rules[e['order']]=e['counts'];staged.update(e['counts'])
canonical=Counter()
for c in rules.values():canonical.update(c)
known.update(canonical);known['universal_constant_arithmetic']+=6
assert known==Counter(S['known_float_named_counts']) and sum(known.values())==142873286 and exact==Counter(S['exact_operation_inventory']);assert canonical==Counter(S['canonical_cold_rule_counts']) and staged==Counter(S['actual_staged_rule_counts']);assert bits==[804,788] and reused==64 and rescues==132 and dict(methods)==S['method_counts'];assert abs(float(1-mass)-S['accuracy'])<2e-16 and float(bounds)==S['covered_conditional_bound_sum'];assert abs(float(l1)-S['covered_absolute_error_sum'])<1e-20
out=dict(exhaustive_saved_cases=len(rows),bitwise_pilot_rows_reused_once=reused,independently_verified_exact_rescues=rescues,methods=dict(methods),accuracy=float(1-mass),reference_accuracy=float(1-refmass),accuracy_difference=float(refmass-mass),conditional_bound_sum=float(bounds),known_float_named_total=sum(known.values()),canonical_rule_arithmetic=canonical['arithmetic'],actual_staged_rule_arithmetic=staged['arithmetic'],rational_max_bits=bits,review_cpu=time.process_time()-start,scope='Whole fixed seed0 error-ray corpus only. Saved aggregates and all132 exact certificates checked, no integration rerun. General floating errors and exact-rational/MP/library internals remain separate; 300-unit goal fails, not a method-family impossibility.')
(H/'positive_h8_full_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
