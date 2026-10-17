from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import json,hashlib,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/capture');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dec(x):
 if isinstance(x,list):
  if len(x)==2 and all(isinstance(a,str) for a in x):return F(int(x[0]),int(x[1]))
  return tuple(dec(a) for a in x)
 if isinstance(x,dict):return {k:dec(v) for k,v in x.items()}
 return x
p=json.loads((P/'trained_protocol.json').read_text());assert p['source_gate']=='b927baa2 fixed four-parent T3 then T4 source gate'
for file,h in p['pins'].items():assert sha(Path(file))==h,file
zero=json.loads((P/'zero_controls.json').read_text());reports=[]
for T in (3,4):
 raw=json.loads((P/f'trained_t{T}.json').read_text());r=dec(raw);assert r['complete'] and r['protocol_sha256']==sha(P/'trained_protocol.json')
 assert r['new_initial_cones']==r['new_mass_reference_MC_calls']==r['observer']['mass_or_final_calls']==0 and r['reused_initial_cones']==4
 wi,wh,wo=r['model'];H=len(wi);d=T-1;K=T-1;N=len(r['states']);assert H==4
 rows=r['analysis']['rows'];assert len(rows)==N==r['observer']['states']
 progress=[json.loads(x) for x in (P/f'trained_t{T}_progress.jsonl').read_text().splitlines()]
 assert [x['row'] for x in progress if x['stage']=='prefix_state']==raw['states']
 assert [x['row'] for x in progress if x['stage']=='signature_result']==raw['analysis']['rows']
 assert len([x for x in progress if x['stage']=='signature_attempt'])==N
 assert [x['parent'] for x in r['parents']]==[0,1,20,21]
 assert sum(x['states'] for x in r['parents'])==N and sum(x['source_proxy'] for x in r['parents'])==r['observer']['proxy']
 V=tuple(tuple(row[i]-wo[0][i] for i in range(H)) for row in wo[1:]);sumcounts=Counter();maxdiff=F(0);equal=0
 for index,(s,a) in enumerate(zip(r['states'],rows,strict=True)):
  assert a['index']==index and a['parent']==s['parent'] and a['truth']==s['truth'] and a['depth']==s['depth']==d
  assert a['history']==s['history'] and len(s['history'])==d and s['g']==s['history'][-1]
  initial=zero['initial_cones'][str(T)][s['parent']];assert s['truth']==initial['truth'] and s['n_true']==tuple(initial['n_true'])
  assert s['history'][0]==tuple(int(initial['sigma']*v>0) for v in wi)
  M=tuple((F(0),)*d for _ in range(H))
  for t,g in enumerate(s['history']):
   M=tuple(tuple(g[i]*(sum(wh[i][k]*M[k][j] for k in range(H))+(wi[i] if j==t else 0)) for j in range(d)) for i in range(H))
  S=tuple(tuple(F(float.fromhex(v)) for v in row[:d]) for row in s['state'])
  assert M==a['exact_state'] and S==a['stored_state']
  for i in range(H):
   for j in range(T):assert F(float.fromhex(s['state'][i][j]))==s['g'][i]*F(float.fromhex(s['A'][i][j]))
   assert F(float.fromhex(s['state'][i][-1]))==0
  diff=max(abs(x-y) for m,n in zip(M,S) for x,y in zip(m,n));assert diff==a['max_map_difference'] and a['map_equal']==(M==S)
  maxdiff=max(maxdiff,diff);equal+=M==S
  for tag,MM,cc in zip(('exact','stored'),(M,S),a['canonical_counts'],strict=True):
   B=tuple(tuple(sum(wh[i][k]*MM[k][j] for k in range(H)) for j in range(d))+(wi[i],) for i in range(H))
   # Independent absolute-value decomposition: ReLU(z)=(z+abs(z))/2.
   lin=tuple(tuple(sum(V[k][i]*B[i][j] for i in range(H))/2 for j in range(T)) for k in range(K));groups=defaultdict(lambda:[F(0)]*K);neg=nonzero=0
   for i,b in enumerate(B):
    alpha=next((z for z in b if z),None)
    if alpha is None:continue
    nonzero+=1;neg+=alpha<0;u=tuple(z/alpha for z in b)
    for k in range(K):groups[u][k]+=abs(alpha)*V[k][i]
   hinges=tuple((u,tuple(c)) for u,c in sorted(groups.items()) if any(c))
   L=tuple(tuple(lin[k][j]-sum(c[k]*u[j]/2 for u,c in hinges) for j in range(T)) for k in range(K));key=(L,hinges)
   assert key==a[tag+'_key']
   scale=next((abs(z) for row in L for z in row if z),None)
   if scale is None:scale=next((abs(z) for _,c in hinges for z in c if z),None)
   projective=key if scale is None else (tuple(tuple(z/scale for z in row) for row in L),tuple((u,tuple(z/scale for z in c)) for u,c in hinges))
   assert projective==a[tag+'_projective']
   expected=Counter(div=nonzero*T+(0 if scale is None else K*T+len(hinges)*K),mul=nonzero*K+neg*K*T,add=nonzero*K+neg*K*T)
   assert dict(expected)==cc;sumcounts.update(expected)
 summary=r['analysis']['summary'];assert summary['stored_maps_equal_exact']==equal and summary['max_map_difference']==maxdiff
 assert dict(sumcounts)==r['analysis']['canonical_counts']
 formation=N*(d*H*d*(H+1)+2*H*H*d)
 assert r['analysis']['formation_counts']==dict(mul=formation,add=formation,sub=N*2*H*K)
 for field,oldgroups in r['analysis']['groups'].items():
  gg=defaultdict(list)
  for i,row in enumerate(rows):gg[(row['truth'],row['depth'],row[field])].append(i)
  assert tuple(tuple(v) for v in gg.values())==oldgroups and summary['group_counts'][field]==len(gg)
  if field in r['analysis']['additional_different_state_groups']:
   sf='exact_state' if field.startswith('exact') else 'stored_state';extra=tuple(tuple(v) for v in gg.values() if len({rows[i][sf] for i in v})>1)
   assert extra==r['analysis']['additional_different_state_groups'][field] and summary['additional_group_counts'][field]==len(extra)
 reports.append(dict(T=T,states=N,groups=summary['group_counts'],extra_groups=summary['additional_group_counts'],stored_exact=equal,max_map_difference=float(maxdiff),lookahead_calls=r['observer']['lookahead_calls'],lookahead_points=r['observer']['lookahead_points'],author_cpu=r['cpu_seconds'],output_sha256=sha(P/f'trained_t{T}.json')))
result=dict(complete=True,phases=reports,review_cpu=time.process_time()-start,scope='Independent exact saved history recurrence, absolute-value function decomposition, projective scaling, groups and dense operation formulas. No engine/model-input/geometry/mass replay. Source rounded cell validity, H/V coverage, generic label equality and merge-cost claims excluded. Peak-bit accounting not independently replayed.')
(O/'t188_trained_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
