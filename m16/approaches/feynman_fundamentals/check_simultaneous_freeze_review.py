"""Independent rational recurrence and strict interior counterexample; scope qualification."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from fractions import Fraction as F
from pathlib import Path
import json,time,math
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_78/m16/approaches/certificate_review');D=json.loads((P/'simultaneous.json').read_text());checks=0

def run(model,x,frozen=False):
 wi,wh,wo=model;h=[F(0)]*3;states=[];pres=[]
 for t,u in enumerate(x):
  z=[wi[i]*u+sum(wh[i][j]*h[j] for j in range(3)) for i in range(3)];h=[z[i]*int(i==2) for i in range(3)] if frozen and t==2 else [max(F(0),v) for v in z];pres.append(z);states.append(h)
 logits=[sum(a*b for a,b in zip(row,h)) for row in wo];return dict(preactivations=pres,states=states,outputs=logits,winner=max(range(3),key=lambda i:logits[i]),margin=logits[2]-max(logits[:2]))
def rational(v):
 if isinstance(v,str):return F(v)
 if isinstance(v,list):return [rational(x) for x in v]
 if isinstance(v,dict):return {k:rational(x) for k,x in v.items()}
 return v
for row in D['cases']:
 model=rational(row['model']);rays=rational(row['rays'])
 for saved,x in zip(row['direct'],rays):
  for label,frozen in [('actual',False),('frozen',True)]:assert run(model,x,frozen)==rational(saved[label]);checks+=1
  assert x[1]>x[2]>x[0]>0 and min(v for z in run(model,x)['preactivations'][:2] for v in z)>0
 for j,form in enumerate(rational(row['final_forms'])):
  lo=hi=F(0)
  for x in rays:
   norm2=sum(v*v for v in x);scale=1<<64;k=math.isqrt((norm2.numerator*scale*scale)//norm2.denominator);l,u=F(k,scale),F(k+1,scale);assert l*l<=norm2<u*u;value=sum(a*b for a,b in zip(form,x));lo+=value/(u if value>=0 else l);hi+=value/(l if value>=0 else u)
  lo/=3;hi/=3;assert [lo,hi]==rational(row['exact_reference_form_intervals'][j]);assert hi<0 if j<2 else lo>0
case=D['cases'][0];model=rational(case['model']);rays=rational(case['rays']);x=[sum(w*r[j] for w,r in zip((1,100,1),rays)) for j in range(3)];actual=run(model,x);frozen=run(model,x,True);allow=actual['preactivations'][-1][:2]
assert actual['winner']==0 and frozen['winner']==2 and actual['margin']<0<frozen['margin'];assert all(0<v<frozen['margin'] for v in allow) and sum(allow)>frozen['margin'];assert case['source_stats']['resolved']==0 and case['source_gates']==[[0.,0.,1.]]
# Original cone crosses an initial nearest-gap chart boundary; retain limited scope.
gaps=[r[1]+r[0]-2*r[2] for r in rays];assert min(gaps)<0<max(gaps)
out=dict(saved_exact_recurrences=checks,strict_positive_ray_combination=[1,100,1],interior_input=list(map(str,x)),actual_margin=str(actual['margin']),frozen_margin=str(frozen['margin']),individual_allowances=list(map(str,allow)),initial_nearest_gap_values=list(map(str,gaps)),initial_half_family_containment=False,review_cpu=time.process_time()-start,scope='Generic cell-rule simultaneous-freeze counterexample has a strict cone-interior witness. Original cone straddles initial nearest-gap charts; actual reachable candidate-cell failure remains a separate stronger control. No trained-panel accuracy refuted.')
(H/'simultaneous_freeze_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
