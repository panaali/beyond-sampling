"""Saved768 exact mask-region algebra, no author/model forward calls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from itertools import permutations,product
import json,hashlib,resource,time,ast
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fading_influence');O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();dot=lambda a,b:sum(x*y for x,y in zip(a,b,strict=True))
tree=ast.parse((O/'check_fading_controls_review.py').read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unpack'];exec(compile(tree,'reviewer unpack','exec'))
def audit(row):
 m=row['model'];w=row['witness'];H,T,D=m['H'],m['T'],m['D'];assert T==4;x=w['x'];wi=[F(z,D) for z in m['wi']];wh=[[F(z,D) for z in r] for r in m['wh']];wo=[[F(z,D) for z in r] for r in m['wo']]
 # Region map from the SAVED mask, checking all defining inequalities at x.
 J=[[F(0)]*T for _ in range(H)];radii=[];zeros=[];boundaries=[]
 for t in range(T):
  B=[[sum(wh[i][k]*J[k][j] for k in range(H))+(wi[i] if j==t else 0) for j in range(T)] for i in range(H)]
  for i,b in enumerate(B):
   v=dot(b,x);assert (v>0)==w['history'][t][i]
   if not any(b):zeros.append([t,i])
   elif v==0:boundaries.append([t,i])
   else:radii.append(abs(v)/(2*sum(map(abs,b))))
  J=[b if w['history'][t][i] else [F(0)]*T for i,b in enumerate(B)]
 L=[[sum(wo[a][i]*J[i][j] for i in range(H)) for j in range(T)] for a in range(T)]
 assert L==[[F(z,w['logit_denominator']) for z in r] for r in w['logit_map_numerators']]
 assert zeros==w['identically_zero_gates'] and boundaries==w['boundary_gates'];gaps=[abs(x[i]-x[j]) for i in range(T) for j in range(i)];strict=not boundaries and all(gaps);assert strict==w['strict']==w['local_derivative_certified']
 if all(gaps):radii.append(F(min(gaps),4))
 assert w['radius']==(min(radii) if strict else None)
 sensitivities=[max(L[a][j] for a in range(T))-min(L[a][j] for a in range(T)) for j in range(T)];assert sensitivities==w['sensitivities_by_input'] and sensitivities[::-1]==w['age_sensitivities']
 u=list(map(abs,wi));upper=[]
 for age,r in enumerate(row['envelope']['rows']):
  expected=[{'a':a,'b':b,'bound':dot([abs(v-z) for v,z in zip(wo[a],wo[b])],u)} for a in range(T) for b in range(a)];assert expected==r['pairs'] and r['maximum']==max(v['bound'] for v in expected);upper.append(r['maximum']);u=[dot(list(map(abs,z)),u) for z in wh]
 assert all(s<=upper[T-1-j] for j,s in enumerate(sensitivities))
 npairs=T*(T-1)//2;nz=H*T-len(zeros)-len(boundaries)
 ec=dict(mul=1+T*npairs*H+(T-1)*(H*H+1),add=T*npairs*H+(T-1)*H*H,sub=T*npairs*H)
 wc=dict(mul=T*(H*T*H+H+H*(H+2)+H*T+1)+nz+T*T*H+T*H+T*T+1,add=T*(H*T*H+H+H*(H+1)+H*T)+nz*T+T*T*H+T*H+T*T,sub=T+npairs)
 assert ec==row['envelope']['counts'] and wc==w['counts'];return wc

outpath=O/'fading_trained_review_results.json';assert not outpath.exists();g=read(P/'trained_protocol.json');d=unpack(read(P/'trained.json'));assert d['complete'] and d['protocol_sha256']==sha(P/'trained_protocol.json')
for p,h in g['pins'].items():assert sha(p)==h,p
assert [r['name'] for r in d['models']]==['H4T4seed0','H8T4seed0']
xs=[[v*s for v,s in zip(p,sign)] for p in permutations((1,2,3,4)) for sign in product((-1,1),repeat=4)]
events=[unpack(json.loads(x)) for x in (P/'trained_progress.jsonl').read_text().splitlines()];assert len(events)==1543
rows=[]
for model in d['models']:
 witnesses=model['witnesses'];assert model['complete'] and len(witnesses)==384 and [w['x'] for w in witnesses]==xs and [w['index'] for w in witnesses]==list(range(384))
 assert [r['witness'] for r in events if r['stage']=='witness_result' and r['name']==model['name']]==witnesses
 counts=Counter()
 for w in witnesses:counts.update(audit({'name':model['name'],'model':model['model'],'envelope':model['envelope'],'witness':w}))
 strict=[w for w in witnesses if w['strict']];s=model['summary'];assert s['strict_witnesses']==len(strict) and s['unresolved_boundary_witnesses']==384-len(strict)
 assert s['integer_named_counts']==dict(counts) and s['max_retained_integer_bits']==max(w['bits'] for w in witnesses)
 assert s['witness_cpu']==sum(w['cpu'] for w in witnesses)
 hist=len({tuple(tuple(g) for g in w['history']) for w in strict});assert hist==s['distinct_strict_gate_histories']
 maximum=[max((w['age_sensitivities'][a] for w in strict),default=None) for a in range(4)];assert maximum==s['max_sensitivity_by_age']
 assert [sum(w['age_sensitivities'][a]==0 for w in strict) for a in range(4)]==s['zero_sensitivity_counts_by_age']
 oldnew=Counter(older_greater=0,equal=0,older_smaller=0,newest_zero=0)
 for w in strict:
  a,b=w['age_sensitivities'][3],w['age_sensitivities'][0];oldnew['older_greater' if a>b else 'equal' if a==b else 'older_smaller']+=1;oldnew['newest_zero']+=b==0
 assert dict(oldnew)==s['oldest_vs_newest_counts']
 rows.append({'model':model['name'],'strict':len(strict),'distinct_histories':hist,'maximum_by_age':list(map(float,maximum)),'oldest_vs_newest':dict(oldnew),'named_integer_counts':dict(counts)})
assert d['process_cpu']>=d['cpu'] and d['process_cpu']<=110
out={'pass':True,'witnesses':768,'rows':rows,'author_process_cpu':d['process_cpu'],'review_cpu':time.process_time()-start,'result_sha256':sha(P/'trained.json'),'protocol_sha256':sha(P/'trained_protocol.json'),'scope':'Independent exact saved-mask affine maps/defining inequalities/radii/envelope and count/summary audit. Checkpoint/source pins bind captured coefficients; no checkpoint reload or author/forward calls. Finite local witnesses are not probability, asymptotic nondecay, or rejection of rank-conditioned forgetting/merging.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
