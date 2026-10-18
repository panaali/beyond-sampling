"""Independent exact projection certificates, completeness and lifted witnesses."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import json,time
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_78/m16/approaches');D=json.loads((P/'prefix_face_projection/pilot.json').read_text());forms=json.loads((P/'joint_strip_continuity/pilot.json').read_text())['proof']['forms'];moving=[i for i,f in enumerate(forms) if F(f['row'][2])];assert {(r['source_index'],r['rank_side']) for r in D['records']}=={(i,s) for i in moving for s in ('01','10')};assert not D['unknown_forms'];assert set(D['zero_state_forms'])==set(range(len(forms)))-set(moving)
count=0;feasible=[];dualrows=0
for r in D['records']:
 boundary=tuple(map(F,r['boundary']));j=2 if abs(boundary[2])>=abs(boundary[1]) else 1;k=3-j;original=[tuple(map(F,q)) for q in r['original_constraints']];assert original[0]==(1,0,0,0)
 rows=[tuple(q[v]-q[j]*boundary[v]/boundary[j] for v in (0,k,3)) for q in original];assert rows==[tuple(map(F,q)) for q in r['substituted_constraints']];p=r['projection'];positive=[i for i,q in enumerate(rows) if q[1]>0];negative=[i for i,q in enumerate(rows) if q[1]<0];zero=[i for i,q in enumerate(rows) if q[1]==0];expected={(i,) for i in zero}|{(i,j) for i in positive for j in negative};seen=set();low=F(0);high=None;bad=[]
 for n,z in enumerate(p['reduced']):
  sources=[(i,F(w)) for i,w in z['sources']];assert all(w>=0 for i,w in sources);seen.add(tuple(i for i,w in sources));combined=tuple(sum(w*rows[i][v] for i,w in sources) for v in range(3));assert combined[1]==0 and (combined[0],combined[2])==tuple(map(F,z['row']));a,c=combined[0],combined[2];dualrows+=1
  if a>0:low=max(low,-c/a)
  elif a<0:high=min(high,-c/a) if high is not None else -c/a
  elif c<0:bad.append(n)
 assert seen==expected and len(seen)==len(p['reduced']);assert low==F(p['lower']);assert high==(None if p['upper'] is None else F(p['upper']));ok=not bad and (high is None or low<=high);assert ok==p['feasible'] and bad==p['contradictions'];count+=1
 if ok:
  s,x,y=map(F,r['witness']);assert sum(a*b for a,b in zip(boundary,(s,x,y,1)))==0 and all(sum(a*b for a,b in zip(q,(s,x,y,1)))>=0 for q in original);feasible.append((low,high));assert r['rank_side']=='10' and forms[r['source_index']]['first']==[0,1]
# Independently join closed projections and compare the only resulting ray.
end=F(0)
for lo,hi in sorted(feasible,key=lambda p:p[0]):
 assert lo<=end or end is None
 if end is None:continue
 end=None if hi is None else max(end,hi)
assert end is None and D['union']==[['0',None]] and D['sufficient_open_invariance_intervals']==[]
out=dict(exact_projection_cases=count,nonnegative_dual_rows=dualrows,feasible_faces=len(feasible),projected_union='[0,infinity)',sufficient_invariance_intervals=[],all_feasible_first_gate=[0,1],all_feasible_rank_side='10',review_cpu=time.process_time()-start,scope='Complete exact one-coordinate elimination certificates and lifted witnesses. No sufficient unbounded-input invariance sector; this is not proof the pair value varies everywhere. Earlier truncated-domain local invariance remains valid.')
(H/'prefix_projection_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
