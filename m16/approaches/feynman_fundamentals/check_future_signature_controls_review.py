"""Independent absolute-value normal form for saved hinge signatures."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict,Counter
import json,hashlib,resource,time,itertools
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence');O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(x):
 if isinstance(x,list):
  if len(x)==2 and all(isinstance(v,str) and v.lstrip('-').isdigit() for v in x):return F(*map(int,x))
  return [unpack(v) for v in x]
 if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
 return x
def tuplify(x):return tuple(map(tuplify,x)) if isinstance(x,list) else x
def normal(net,canon):
 B,V=net['B'],net['V'];K=len(V);d=len(B[0]) if B else len(net['L'][0]);linear=net['L'] or [[F(0)]*d for _ in range(K)]
 # ReLU(s)=(s+|s|)/2, orientation-free signed linear component.
 A=[[linear[k][j]+sum((V[k][i]*b[j]/2 for i,b in enumerate(B)),F(0)) for j in range(d)] for k in range(K)]
 groups=defaultdict(lambda:[F(0)]*K);neg=nonzero=0
 for i,b in enumerate(B):
  if not any(b):continue
  alpha=next(x for x in b if x);u=tuple(x/alpha for x in b);nonzero+=1;neg+=alpha<0
  for k in range(K):groups[u][k]+=abs(alpha)*V[k][i]
 expected=tuple((u,tuple(c)) for u,c in sorted(groups.items()) if any(c));L,hinges=tuplify(canon['key']);assert hinges==expected
 assert [[L[k][j]+sum((c[k]*u[j]/2 for u,c in hinges),F(0)) for j in range(d)] for k in range(K)]==A
 first=next((abs(x) for row in L for x in row if x),None) or next((abs(x) for _,c in hinges for x in c if x),None)
 proj=(tuple(tuple(x/first for x in row) for row in L),tuple((u,tuple(x/first for x in c)) for u,c in hinges)) if first else (L,hinges)
 assert proj==tuplify(canon['projective'])
 expectedcount=Counter(div=nonzero*d+(K*d+len(hinges)*K if first else 0),mul=nonzero*K+neg*K*d,add=nonzero*K+neg*K*d)
 assert +expectedcount==+Counter(canon['counts'])
 return (L,hinges),proj,K,d
outpath=O/'future_signature_controls_review_results.json';assert not outpath.exists();g=read(P/'protocol.json');d=unpack(read(P/'controls.json'))
for p,h in g['pins'].items():assert sha(p)==h
assert d['complete'] and d['all_pass'] and len(d['rows'])==24 and len(d['guards'])==4
progress=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()];assert len(progress)==56 and [unpack(x['row']) for x in progress if x['stage']=='result']==d['rows']
counts=Counter();calls=0;equal=projective=0
for r in d['rows']:
 a,pa,K,dim=normal(r['a'],r['canonical_a']);b,pb,Kb,dim2=normal(r['b'],r['canonical_b']);assert K==Kb and dim==dim2
 assert r['exact_equal']==r['expected']['exact']==(a==b) and r['projective_equal']==r['expected']['projective']==(pa==pb) and r['passed'] and r['function_checks']
 equal+=a==b;projective+=pa==pb;calls+=2*3**dim
 if r['expected']['labels'] is not None:assert r['fixed_grid_label_equal']==r['expected']['labels']
 if a!=b:
  w=r['difference_witness'];x=w['x']
  for side in ('a','b'):
   n=r[side];L=n['L'] or [[F(0)]*dim for _ in range(K)];values=[sum((coef*max(F(0),sum(u*v for u,v in zip(form,x))) for coef,form in zip(n['V'][k],n['B'])),F(0))+sum(u*v for u,v in zip(L[k],x)) for k in range(K)]
   assert values==w[side]
  assert w['a']!=w['b']
 else:assert r['difference_witness'] is None
 counts.update(r['canonical_a']['counts']);counts.update(r['canonical_b']['counts'])
assert calls==d['synthetic_direct_evaluations']==564 and all(r['passed'] for r in d['guards'])
assert {r['id'] for r in d['guards']}=={'bad_form_width','bad_coeff_width','no_outputs','bad_linear_width'}
out={'pass':True,'pairs':24,'guards':4,'exact_equal_pairs':equal,'positive_projective_equal_pairs':projective,'synthetic_evaluations_retained':calls,'named_rational_counts':dict(counts),'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'result_sha256':sha(P/'controls.json'),'protocol_sha256':sha(P/'protocol.json'),'scope':'Independent absolute-value normal form and saved difference-witness/source-category audit. No author canonical/grid replay or trained capture. Exact global margin equivalence is stronger than label/prefix-domain equivalence; nonmatches do not disprove useful merging.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
