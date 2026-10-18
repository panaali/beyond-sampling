"""Saved zero-prefix trace and initial-cone binding; no geometry/engine replay."""
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import json,hashlib,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/capture');O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();arr=lambda a:[[F(float.fromhex(x)) for x in row] for row in a];dot=lambda a,b:sum(x*y for x,y in zip(a,b,strict=True))
outpath=O/'future_zero_capture_review_results.json';assert not outpath.exists();g=read(P/'protocol.json');d=read(P/'zero_controls.json');events=[json.loads(x) for x in (P/'zero_progress.jsonl').read_text().splitlines()]
for p,h in g['pins'].items():assert sha(p)==h,p
assert d['complete'] and d['all_pass'] and len(d['rows'])==2 and len(events)==367
states=[r['row'] for r in events if r['stage']=='prefix_state'];assert len(states)==120
proxy=0;initial=0
for row in d['rows']:
 T=row['T'];count=24 if T==3 else 96;halves=d['initial_cones'][str(T)];assert len(halves)==count and row['passed'];selected=[r for r in states if r['depth']==T-1];assert len(selected)==count
 initial_event=next(r for r in events if r['stage']=='initial_cones_result' and r['T']==T);initial+=initial_event['source_proxy']
 E=[[F(i==j) for j in range(T)] for i in range(T)];keys=[(sg,p,half) for sg in (1,-1) for p in permutations(range(T)) for half in (1,-1)]
 for i,(h,state,parent,key) in enumerate(zip(halves,selected,row['parents'],keys,strict=True)):
  sg,perm,half=key;a,b,c=perm[:3];assert h['sigma']==sg and tuple(h['perm'])==perm and h['truth']==b
  ntrue=[E[a][j]-E[b][j] if half==1 else E[b][j]-E[c][j] for j in range(T)];assert list(map(lambda x:F(float.fromhex(x)),h['n_true']))==ntrue
  cone=h['cone'];R,N=arr(cone['R']),arr(cone['N']);tags=cone['T'];assert len(R)==len(tags) and all(len(r)==T for r in R+N) and all(len(t)==len(N) for t in tags)
  ordering=[[E[perm[k]][j]-E[perm[k+1]][j] for j in range(T)] for k in range(T-1)];base=ordering+[[sg*x for x in E[0]]];split=[half*(2*E[b][j]-E[a][j]-E[c][j]) for j in range(T)]
  assert N==base or N==base+[split]
  for ray,tag in zip(R,tags,strict=True):
   assert abs(dot(ray,ray)-1)<F(1,10**14) and all(dot(n,ray)>=-F(1,10**14) for n in base+[split])
   for n,on in zip(N,tag,strict=True):assert (abs(dot(n,ray))<F(1,10**14))==on
  assert state['parent']==parent['parent']==i and state['truth']==b and state['cone']==cone and state['n_true']==h['n_true'] and state['depth']==T-1
  assert len(state['history'])==T-1 and all(g==[0]*4 for g in state['history']) and state['g']==[0]*4
  assert arr(state['A'])==arr(state['state'])==[[0]*T for _ in range(4)]
  expected=5*(T+1)*len(R)*(T-2);assert parent['states']==1 and parent['source_proxy']==expected;proxy+=expected
 assert all(v for v in row['checks'].values() if isinstance(v,bool))
obs=d['observer'];assert obs['states']==120 and obs['finish_calls']==336 and obs['lookahead_calls']==obs['lookahead_points']==obs['mass_or_final_calls']==0 and obs['proxy']==proxy+initial==81864
out={'pass':True,'states':120,'finish_entries':336,'lookahead_or_mass_calls':0,'prefix_proxy':proxy,'initial_cone_proxy':initial,'incomplete_source_proxy':obs['proxy'],'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'result_sha256':sha(P/'zero_controls.json'),'progress_sha256':sha(P/'zero_progress.jsonl'),'protocol_sha256':sha(P/'protocol.json'),'scope':'Saved chart ordering/truth/facet incidence diagnostics, zero-state/history/depth and source prefix proxy verified. Initial geometric predicates use finite saved-float tolerance, no coverage certificate or geometry/engine replay. Setup/wrapper/full native costs and trained frozen-history validity remain open.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
