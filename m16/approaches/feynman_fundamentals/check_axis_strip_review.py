"""Independent exact pairwise intersections for all32 word feasibility decisions."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json,time
start=time.process_time();here=Path(__file__).parent;root=Path('/workspace/work/recover-small_h2_78/m16/approaches');d=json.loads((root/'axis_strip_continuity/pilot.json').read_text());previous=json.loads((root/'joint_strip_continuity/pilot.json').read_text());forms=previous['proof']['forms'];results=[];checked=0
for row in d['rows']:
 axis=row['axis'];lo,hi=map(F,row['hidden_interval']);xl,xh=map(F,row['x_limits']);box=[(F(1),F(0),-lo),(F(-1),F(0),hi),(F(0),F(1),-xl),(F(0),F(-1),xh)];allowed=set()
 for word in row['words']:
  constraints=box+[tuple(map(F,q)) for q in word['constraints']];vertices=set()
  for (a,b,c),(e,f,g) in combinations(constraints,2):
   det=a*f-b*e
   if not det:continue
   s=(b*g-c*f)/det;x=(c*e-a*g)/det
   if all(A*s+B*x+C>=0 for A,B,C in constraints):vertices.add((s,x))
  assert bool(vertices)==word['feasible'];checked+=1
  for vertex in word['vertices']:
   s,x=map(F,vertex);assert all(A*s+B*x+C>=0 for A,B,C in constraints)
  if vertices:allowed.add((tuple(word['first']),tuple(word['middle'])))
 def coefficient(allowed):
  unique={}
  for form in forms:
   if allowed is not None and (tuple(form['first']),tuple(form['middle'])) not in allowed:continue
   A,B,C0,C1,E=map(F,form['row']);C=[C0,C1][axis]
   if not C:continue
   normal=max(abs(A),abs(B));assert normal;r=(A,B,C,E);pivot=next(v for v in r if v);key=tuple(v/pivot for v in r);value=abs(C)/normal
   if key in unique:assert unique[key]==value
   unique[key]=value
  return sum(unique.values(),F(0))
 A=coefficient(None);K=coefficient(allowed);assert A==F(row['all_word_bound']['K_unique']) and K==F(row['feasible_bound']['K_unique']);pl,ph=map(F,row['prefix_interval']);scale=(hi-lo)/(ph-pl);cert=F(81,2)*K*scale;assert cert==F(row['elementary_prefix_coefficient']);display=float(K*scale)/previous['display']['joint_rank_masses'][1];assert abs(display-row['display']['need1_prefix_CDF_bound'])<1e-13
 results.append(dict(axis=axis,feasible_words=len(allowed),all_word_axis_K=float(A),feasible_axis_K=float(K),prefix_CDF_display=display,elementary_prefix_coefficient=float(cert)))
out=dict(independent_pairwise_feasibility_cases=checked,rows=results,review_cpu=time.process_time()-start,scope='Exact closed-set vertex enumeration independent of clipping; local frozen axis neighborhoods, no later mixed-support interpolation, population accuracy or practical-grid certificate.')
(here/'axis_strip_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
