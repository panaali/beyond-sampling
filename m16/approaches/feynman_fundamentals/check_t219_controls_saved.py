"""Independent small exact H-cone/minor and closed-history saved control review.
No author imports, clipping/extrusion calls, target weights or forward evaluation.
"""
from pathlib import Path
from fractions import Fraction
from itertools import combinations, permutations, product
from collections import Counter
from math import gcd
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_34/review/target_prefix_recount')
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def primitive(v):
 g=0
 for x in v:g=gcd(g,abs(x))
 assert g
 return tuple(x//g for x in v)
def dot(a,b):return sum(x*y for x,y in zip(a,b,strict=True))
def det(a):
 n=len(a);s=0
 for p in permutations(range(n)):
  t=1
  for i in range(n):t*=a[i][p[i]]
  s+=(-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))*t
 return s
def rank(a):
 if not a:return 0
 for n in range(min(len(a),len(a[0])),0,-1):
  if any(det([[a[i][j] for j in cs] for i in rs]) for rs in combinations(range(len(a)),n) for cs in combinations(range(len(a[0])),n)):return n
 return 0
def rays_of(normals,d):
 # Exhaust every d-1 active-normal intersection, then retain both feasible signs.
 ns=sorted({primitive(v) for v in normals if any(v)})
 out=set()
 for chosen in combinations(ns,d-1):
  v=tuple((-1)**j*det([[r[k] for k in range(d) if k!=j] for r in chosen]) for j in range(d))
  if not any(v):continue
  for s in (-1,1):
   z=primitive(tuple(s*x for x in v))
   if all(dot(n,z)>=0 for n in ns):out.add(z)
 return out

def cone_check(c,witness=None):
 rr=[tuple(v) for v in c['rays']];nn=[tuple(v) for v in c['normals']];d=len(rr[0])
 assert all(type(x) is int for v in rr+nn for x in v)
 assert len(rr)==len(set(rr)) and len(nn)==len(set(nn))
 assert all(len(v)==d and primitive(v)==v for v in rr+nn)
 assert rank(rr)==rank(nn)==d
 assert rays_of(nn,d)==set(rr)
 for r in rr:
  values=[dot(n,r) for n in nn];assert min(values)>=0
  assert rank([n for n,v in zip(nn,values) if v==0])==d-1
 w=primitive(tuple(sum(r[j] for r in rr) for j in range(d)))
 assert all(dot(n,w)>0 for n in nn)
 if witness is not None:assert list(w)==witness
 return w

def history_forms(case,history):
 wi=case['wi'];W=case['wh'];D=case['D'];h=len(wi);d=len(history)
 # Each column is a masked product D_t W ... D_(j+1) W D_j Wi * D^j.
 def column(j,end):
  v=[history[j][i]*wi[i]*D**j for i in range(h)]
  for t in range(j+1,end+1):v=[history[t][i]*dot(W[i],v) for i in range(h)]
  return v
 hidden=[[column(j,d-1)[i] for j in range(d)] for i in range(h)]
 signed=[];zero=0
 for t,mask in enumerate(history):
  for i,on in enumerate(mask):
   b=[dot(W[i],column(j,t-1)) for j in range(t)]+[wi[i]*D**t]+[0]*(d-t-1)
   if any(b):signed.append(tuple(x if on else -x for x in b))
   else:
    if on:return None
    zero+=1
 return hidden,signed,zero

def all_histories(case,d):
 # Exhaust only fixed synthetic words, no author's traversal or clipping routines.
 h=len(case['wi']);out={}
 for bits in product((0,1),repeat=h*d):
  hist=tuple(tuple(bits[t*h:(t+1)*h]) for t in range(d));f=history_forms(case,hist)
  if f is None:continue
  hidden,ns,z=f;rr=rays_of(ns,d)
  if rank(list(rr))!=d:continue
  w=tuple(sum(r[j] for r in rr) for j in range(d))
  if not all(dot(n,w)>0 for n in ns):continue
  out[hist]=(hidden,rr)
 return out

def main():
 start=time.process_time();raw=json.loads((P/'controls_result.json').read_text());p=json.loads((P/'control_protocol.json').read_text());events=[json.loads(x) for x in (P/'controls_progress.jsonl').read_text().splitlines()]
 assert raw['status']=='complete' and raw['protocol_sha256']==sha(P/'control_protocol.json')
 for n,h in p['pins'].items():assert sha(P/n)==h
 g=json.loads((P/'control_gate.json').read_text());assert raw['gate']==g and g['clearance'] and g['protocol_sha256']==sha(P/'control_protocol.json') and sha(g['review_artifact'])==g['review_sha256']
 assert len(raw['rows'])==len(p['cases'])==18 and len(events)==36
 assert events==[e for c in p['cases'] for e in [dict(phase='attempt',id=c['id']),dict(phase='result',id=c['id'],status='pass')]]
 assert [raw[k] for k in ['model_calls','mass_calls','random_calls']]==[0,0,0]
 assert raw['resource_limits']['address_space'][0]<=2**31 and raw['resource_limits']['cpu'][0]<=20
 total=Counter();states=0;cones=0;guards=0;invariants=0;counts={};seen={};maxbits=0
 for row,case in zip(raw['rows'],p['cases'],strict=True):
  assert row['case']==case and row['id']==case['id'] and row['status']=='pass'
  assert set(row['counts'])=={'add','sub','mul','integer_div','gcd','rank_calls','clips','lifts'} and all(type(v) is int and v>=0 for v in row['counts'].values())
  total.update(row['counts']);maxbits=max(maxbits,row['max_bits'])
  kind=case['kind']
  if kind in ('clip','lift'):
   cone_check(row['input_cone']);assert row['expected_rays']==case['expected_rays'] and row['rays_pass']
   for side,(child,expected,w) in enumerate(zip(row['children'],case['expected_rays'],row['certificates'],strict=True)):
    if expected is None:assert child is None and w is None;continue
    cone_check(child,w);cones+=1;assert set(map(tuple,child['rays']))==set(map(tuple,expected))
    n=case['form'];ns=row['input_cone']['normals'] if kind=='clip' else [v+[0] for v in row['input_cone']['normals']]
    ns=ns+[[x if side==0 else -x for x in n]]
    assert rays_of(ns,len(n))==set(map(tuple,expected))
  elif kind=='invalid':
   assert row['rejected'] and row['exception'].startswith('AssertionError(') and not any(row['counts'].values());guards+=1
   assert case['id'] in ['zero_input','zero_denominator','bad_shape']
  else:
   assert len(row['depths'])==len(case['counts']);counts[case['id']]=[]
   for level,expected in zip(row['depths'],case['counts'],strict=True):
    d=level['depth'];ss=level['states'];cc=level['certificates'];assert level['count']==level['expected_count']==level['unique_histories']==len(ss)==len(cc)==expected and level['count_pass']
    exhaustive=all_histories(case,d);assert len(exhaustive)==expected
    saved=set()
    for s,c in zip(ss,cc,strict=True):
     hist=tuple(map(tuple,s['history']));assert hist not in saved;saved.add(hist)
     w=cone_check(s,c['witness']);cones+=1;states+=1
     hidden,ns,z=history_forms(case,hist);assert hidden==s['hidden'] and c['zero_forms']==z and c['strict_forms']==len(ns)
     assert all(dot(n,w)>0 for n in ns)
     assert exhaustive[hist]==(hidden,set(map(tuple,s['rays'])))
    assert saved==set(exhaustive);counts[case['id']].append(len(saved))
   if 'base' in case:
    base=seen[case['base']];assert len(row['invariance_checks'])==sum(case['counts']) and all(row['invariance_checks'])
    for level,old in zip(row['depths'],base['depths'],strict=True):
     bm={tuple(map(tuple,s['history'])):s for s in old['states']}
     for s in level['states']:
      b=bm[tuple(map(tuple,s['history']))];scale=2 if case['invariance']=='input_scale2' else -1
      assert s['hidden']==[[scale*x for x in v] for v in b['hidden']]
      rscale=1 if scale==2 else -1
      assert set(map(tuple,s['rays']))=={tuple(rscale*x for x in v) for v in b['rays']};invariants+=1
  seen[case['id']]=row
 assert states==94 and guards==3
 out=dict(complete=True,pass_all=True,source_result_commit='fef35856',pins={n:sha(P/n) for n in ['controls_result.json','controls_progress.jsonl','control_protocol.json','control_gate.json','core.py','controls.py','PROOF.md']},checker_sha256=sha(__file__),cases=18,primitive_cases=7,network_cases=8,guard_cases=3,network_states=states,validated_cones=cones,invariance_states=invariants,progress_events=len(events),counts=counts,reported_ledger_sum=dict(total),reported_max_bits=maxbits,author_cpu=raw['cpu_seconds'],review_cpu=time.process_time()-start,new_author_geometry_model_calls=0,scope='Independent exact determinant/minor H-cone extreme-ray completeness in dimensions1..3, all fixed synthetic gate words, closed-column masked products, strict witnesses, analytic counts, invariances and bindings. Named ledger rows aggregated and checked nonnegative only; internal operation/peak-bit replay is not claimed. Guard records reviewed without executing invalid author paths. No target model count, minimal-affine-region, performance or T10 extrapolation conclusion.')
 (H/'t219_controls_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
