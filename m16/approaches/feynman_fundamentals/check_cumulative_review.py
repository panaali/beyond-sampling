"""Independent Fraction recurrence audit of the proposed cumulative certificate."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import importlib.util,json,time
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/certificate_review')
spec=importlib.util.spec_from_file_location('candidate',P/'cumulative.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
start=time.process_time()
wi=list(map(F,['1','-1/2','1/3']))
wh=[list(map(F,r)) for r in [('1/2','-1/4','3/4'),('-1/2','1/4','1/8'),('1/4','-3/4','1/2')]]
wo=[list(map(F,r)) for r in [('1','-1','1/2'),('-1/3','2','0'),('0','-1/2','1'),('1/4','0','-1')]]
xs=[list(map(F,r)) for r in [(1,2,3,4),(-1,2,-3,4),(1,-2,3,-4),(-2,-1,1,2),(2,1,-1,-2),('1/3','2/3','-1/3','4/3')]]
def run(x,at=-1,g=()):
 h=[F(0)]*3;zs=[]
 for t,u in enumerate(x):
  z=[wi[i]*u+sum(wh[i][j]*h[j] for j in range(3)) for i in range(3)];zs.append(z)
  h=[z[i]*g[i] if t==at else max(z[i],0) for i in range(3)]
 return zs,[sum(a*b for a,b in zip(row,h)) for row in wo]
counts=dict(correct=0,wrong=0,unknown=0);n=0
for x in xs:
 zs,actual=run(x);winner=max(range(4),key=lambda j:actual[j])
 for at in range(4):
  for g in product((0,1),repeat=3):
   _,frozen=run(x,at,g)
   for truth in range(4):
    # Reverse propagation of absolute readout contrast, unlike candidate forward propagation.
    e=[]
    for row in wo:
     v=[abs(a-b) for a,b in zip(wo[truth],row)]
     for _ in range(3-at):v=[sum(v[i]*abs(wh[i][j]) for i in range(3)) for j in range(3)]
     e.append(sum(v[j]*(max(zs[at][j],0)-g[j]*zs[at][j]) for j in range(3)))
    assert e==mod.allowance(wh,wo,zs[at],g,truth,3-at)
    m=[frozen[truth]-v for v in frozen]
    assert all(abs((actual[truth]-actual[j])-m[j])<=e[j] for j in range(4))
    status=mod.decision(m,e,truth);counts[status]+=1;n+=1
    assert status=='unknown' or (status=='correct')==(winner==truth)
saved=json.loads((P/'cumulative_controls.json').read_text());assert counts==saved['status_counts'];assert n==saved['pointwise_label_checks']==768
# Direct conic combinations check the all-ray extension without reusing candidate statuses.
conechecks=0
for file in ['simultaneous.json','family_control.json']:
 data=json.loads((P/file).read_text());cases=data.get('cases',[data])
 for case in cases:
  if 'final_forms' not in case:continue
  A=[list(map(F,r)) for r in case['final_forms']];W=[list(map(F,r)) for r in case['model'][2]];rays=[list(map(F,r)) for r in case['rays']]
  result=mod.final_cell(A,W,(0,0,1),2,rays)['status']
  for coeffs in product((0,1,3),repeat=len(rays)):
   if not any(coeffs):continue
   x=[sum(c*r[j] for c,r in zip(coeffs,rays)) for j in range(len(rays[0]))]
   hidden=[max(sum(a*b for a,b in zip(row,x)),0) for row in A]
   out=[sum(a*b for a,b in zip(row,hidden)) for row in W];winner=max(range(len(W)),key=lambda j:out[j])
   assert result=='unknown' or (result=='correct')==(winner==2);conechecks+=1
out=dict(pointwise_checks=n,status_counts=counts,independent_reverse_allowances=n,nonzero_conic_combination_checks=conechecks,cpu_seconds=time.process_time()-start,scope='Exact finite controls and source proof. Final-cell theorem requires nonempty generators and nonzero conic points; earlier recurrent stages require additional partition or global bounds. No integration into estimator.')
Path(__file__).with_name('cumulative_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
