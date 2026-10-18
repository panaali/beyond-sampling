"""Exact recurrent subcone witness within one initial half-family."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import json,math,time
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_78/m16/approaches/certificate_review');D=json.loads((P/'family_control.json').read_text());wi=list(map(F,D['model'][0]));wh=[list(map(F,r)) for r in D['model'][1]];wo=[list(map(F,r)) for r in D['model'][2]];rays=[tuple(map(F,r)) for r in D['rays']];constraints=[tuple(map(F,r)) for r in D['initial_half_family_constraints']]
def solve(x,freeze=False):
 h=[F(0)]*3;pre=[]
 for t,u in enumerate(x):
  z=[wi[i]*u+sum(w*h[j] for j,w in enumerate(wh[i])) for i in range(3)];pre.append(z);h=[z[i]*int(i==2) for i in range(3)] if freeze and t==2 else [max(v,F(0)) for v in z]
 out=[sum(v*w for v,w in zip(h,row)) for row in wo];return out,pre
for x,r in zip(rays,D['direct']):
 assert all(sum(a*b for a,b in zip(c,x))>0 for c in constraints)
 for label,freeze in [('actual',False),('frozen',True)]:
  out,pre=solve(x,freeze);assert out==list(map(F,r[label]['outputs'])) and pre==[list(map(F,z)) for z in r[label]['preactivations']];assert min(v for z in pre[:2] for v in z)>0
for j,form in enumerate(D['final_forms']):
 row=list(map(F,form));low=high=F(0)
 for x in rays:
  q=sum(v*v for v in x);scale=1<<64;k=math.isqrt((q.numerator*scale*scale)//q.denominator);lo,hi=F(k,scale),F(k+1,scale);v=sum(a*b for a,b in zip(row,x));low+=v/(hi if v>=0 else lo);high+=v/(lo if v>=0 else hi)
 assert [low/3,high/3]==list(map(F,D['reference_form_intervals'][j]));assert high<0 if j<2 else low>0
det=sum(((-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))*math.prod(rays[i][p[i]] for i in range(3)) for p in permutations(range(3))),F(0));assert det
x=[sum(w*r[j] for w,r in zip((1,100,1),rays)) for j in range(3)];actual,pre=solve(x);frozen,_=solve(x,True);am=actual[2]-max(actual[:2]);fm=frozen[2]-max(frozen[:2]);allow=pre[-1][:2];assert am<0<fm and max(allow)<fm<sum(allow) and all(sum(a*b for a,b in zip(c,x))>0 for c in constraints)
assert D['source_stats']['cut']==2 and D['source_stats']['resolved']==0 and D['source_gates']==[0.,0.,1.]
out=dict(independent_recurrences=8,exact_ray_determinant=str(det),interior_positive_ray_weights=[1,100,1],interior_input=list(map(str,x)),actual_margin=str(am),frozen_margin=str(fm),individual_allowances=list(map(str,allow)),all_initial_half_slacks_strict=True,review_cpu=time.process_time()-start,scope='Soundness counterexample for generic rule on a valid recurrent subcone within one initial family. Does not prove the default full solver visits this exact subcone or refute saved trained accuracy.')
(H/'half_family_freeze_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
