"""Independent three-variable recurrence and pairwise closed-face enumeration."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json,time,torch
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_78/m16/approaches');D=json.loads((P/'face_strip_continuity/pilot.json').read_text());prior=json.loads((P/'axis_strip_continuity/pilot.json').read_text());J=json.loads((P/'joint_strip_continuity/pilot.json').read_text());forms=J['proof']['forms'];sd=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',weights_only=True,map_location='cpu');wi=[F(float(v)) for v in sd['rnn.weight_ih_l0'][:,0]];wh=[[F(float(v)) for v in r] for r in sd['rnn.weight_hh_l0']];wo=[[F(float(v)) for v in r] for r in sd['linear.weight']]
def step(h,x):return [tuple(sum(wh[i][k]*h[k][j] for k in range(2))+wi[i]*x[j] for j in range(4)) for i in range(2)]
def constraints(form,axis,side):
 h=[(F(int(i==axis)),F(0),F(0),F(0)) for i in range(2)];rows=[]
 for x,mask in [((0,1,0,0),form['first']),((0,0,0,1),form['middle'])]:
  z=step(h,x);rows.extend(tuple((1 if on else -1)*v for v in r) for on,r in zip(mask,z));h=[tuple(int(on)*v for v in r) for on,r in zip(mask,z)]
 rows.extend([(0,-1 if side==0 else 1,0,1 if side==0 else -1),(0,0,1 if side==0 else -1,-1 if side==0 else 1)])
 z=step(h,(0,0,1,0))
 if form['kind']=='margin':
  mask=form['final'];rows.extend(tuple((1 if on else -1)*v for v in r) for on,r in zip(mask,z))
  for other in range(4):
   if other!=2:rows.append(tuple(sum((wo[2][i]-wo[other][i])*int(mask[i])*z[i][j] for i in range(2)) for j in range(4)))
 return rows,z
checked=0;outrows=[]
for row,old in zip(D['rows'],prior['rows']):
 axis=row['axis'];lo,hi=map(F,row['hidden_interval']);total=F(0);damped=F(0);allK=F(0);only=F(0);feasible=0;degenerate=0;kept=0
 assert len(row['records'])==len(old['feasible_bound']['unique_records'])
 for rec,oldrec in zip(row['records'],old['feasible_bound']['unique_records']):
  A,B,C,E=map(F,rec['row']);coef=abs(C)/max(abs(A),abs(B));assert coef==F(rec['coefficient']);allK+=coef
  vals=[C*lo+E,C*hi+E];offset=F(0) if min(vals)<=0<=max(vals) else min(map(abs,vals));q=offset**2/(2*(A*A+B*B));factor=1/(1+q);assert factor==F(rec['density']['factor']) and q==F(rec['density']['q']);only+=coef*factor
  assert {(o['source_index'],o['rank_side']) for o in rec['occurrences']}=={(i,s) for i in oldrec['source_indices'] for s in ('01','10')}
  keep=False
  for occ in rec['occurrences']:
   form=forms[occ['source_index']];raw,z=constraints(form,axis,0 if occ['rank_side']=='01' else 1);assert raw==[tuple(map(F,r)) for r in occ['original_constraints']]
   boundary=tuple(map(F,occ['boundary']));raw=[(0,1,0,6),(0,-1,0,6),(0,0,1,6),(0,0,-1,6)]+raw
   j=2 if abs(boundary[2])>=abs(boundary[1]) else 1;k=3-j
   reduced=[tuple(F(r[v])-F(r[j])*boundary[v]/boundary[j] for v in (0,k,3)) for r in raw]
   assert reduced==[tuple(map(F,r)) for r in occ['constraints']]
   rows=reduced+[(1,0,-lo),(-1,0,hi),(0,1,6),(0,-1,6)];vertices=set()
   for (a,b,c),(d,e,f) in combinations(rows,2):
    det=a*e-b*d
    if not det:continue
    s=F(b*f-c*e)/det;t=F(c*d-a*f)/det
    if all(u*s+v*t+w>=0 for u,v,w in rows):vertices.add((s,t))
   assert bool(vertices)==occ['feasible'];assert set(tuple(map(F,v)) for v in occ['vertices'])==vertices
   # A bounded nonempty convex set is degenerate iff its extreme vertices are collinear.
   degen=False
   if vertices:
    a=next(iter(vertices));degen=all((b[0]-a[0])*(c[1]-a[1])==(b[1]-a[1])*(c[0]-a[0]) for b,c in combinations(vertices,2));assert degen==(F(occ['area'])==0)
   feasible+=bool(vertices);degenerate+=degen;checked+=1;keep|=bool(vertices)
  assert keep==rec['keep'];kept+=keep
  if keep:total+=coef;damped+=coef*factor
 assert (allK,total,only,damped)==tuple(F(row[k]) for k in ('K_word','K_face','K_density_only','K_face_density'))
 elementary=F(81,2)*abs(wi[axis])*damped;assert elementary==F(row['elementary_prefix']);assert (feasible,degenerate,kept)==(row['feasible_occurrences'],row['degenerate_feasible_occurrences'],row['unique_after'])
 outrows.append(dict(axis=axis,unique_kept=kept,feasible_occurrences=feasible,degenerate_retained=degenerate,face_K=float(total),damped_K=float(damped),elementary_prefix=float(elementary)))
assert checked==D['face_problems']==88
out=dict(independent_exact_faces=checked,rows=outrows,review_cpu=time.process_time()-start,scope='Independent forced affine recurrence and all-pair closed-face vertices, exact coefficient/damping sums. Axis0 zero proves invariance only on frozen local interval and truncated rank domain, not global correctness/accuracy.')
(H/'face_strip_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
