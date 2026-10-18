"""Stored interval audit plus independent Leibniz exact determinant subset."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import json,time,math,hashlib
import numpy as np
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_56/m16/approaches/determinant4_interval');summary=json.loads((src/'summary.json').read_text());rows=[]
for k in range(4):rows+=json.loads((src/f'corpus_{k}.json').read_text())['rows']
assert [r['index'] for r in rows]==list(range(6347));better=worse=equal=0;widths=[];worst={k:(F(0),None) for k in ('new47','old95')}
for r in rows:
 d=F(r['exact_determinant']);v=r['interval'];lo,hi=F(v['lo']),F(v['hi']);assert lo<=d<=hi and lo*hi>0;assert v['contains_exact'];assert v['counts']==dict(arithmetic=154,comparisons=243,finite_checks=340,nextafter=94);widths.append((hi-lo)/abs(d));errors=[]
 for key,count in [('new47',47),('old95',95)]:
  p=r['point'][key];e=abs(F(p['value'])-d);assert math.isclose(float(e),p['absolute_error'],rel_tol=1e-14,abs_tol=1e-300);assert p['counts']['arithmetic']==count;errors.append(e)
  rel=e/abs(d)
  if rel>worst[key][0]:worst[key]=(rel,r['index'])
 better+=errors[0]<errors[1];worse+=errors[0]>errors[1];equal+=errors[0]==errors[1]
assert (better,worse,equal)==(3792,1701,854)
fixed={0,2115,4231,6346,3222,3219,3221,3224,5962,3251,3254,3836,2567,2577,5448,2020,4707,4730};fixed.update(range(0,6347,127));fixed.add(max(range(6347),key=widths.__getitem__));rays=np.load('/workspace/work/recover-small_h2_34/m16/approaches/class_region_t4/quadratic_exact_mass_calls.npz')['rays'];ps=list(permutations(range(4)))
for i in sorted(fixed):
 R=[[F(float(x)) for x in row] for row in rays[i]];value=F(0)
 for p in ps:
  sign=(-1)**sum(p[a]>p[b] for a in range(4) for b in range(a+1,4));value+=sign*math.prod(R[a][p[a]] for a in range(4))
 assert value==F(rows[i]['exact_determinant'])
for k,(v,i) in worst.items():assert i==summary['point'][k]['worst_relative_index'];assert float(v)==summary['point'][k]['maximum_absolute_relative_error']
out=dict(all_intervals=6347,independent_exact_Leibniz_subset=len(fixed),point_better=better,point_worse=worse,point_equal=equal,max_relative_interval_width=float(max(widths)),worst_relative={k:dict(index=i,error=float(v)) for k,(v,i) in worst.items()},review_cpu=time.process_time()-start,scope='Independent stored-float interval containment and exact determinant subset; interval validity assumes ordinary correctly-rounded binary64 operations and nextafter. No geometry/mass certificate, universal point accuracy gain or complete hardware cost.')
(here/'determinant_interval_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
