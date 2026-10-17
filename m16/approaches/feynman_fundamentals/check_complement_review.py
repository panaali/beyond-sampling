"""Independent exact affine and all-box checks using saved model, no author imports."""
import json,time
from pathlib import Path
from fractions import Fraction as F
from itertools import product
start=time.process_time();p=Path('/workspace/work/recover-small_h4_4/m16/approaches/output_forgetting_complement/pilot.json');d=json.loads(p.read_text())
wi=tuple(map(F,d['model'][0]));wh=[list(map(F,r)) for r in d['model'][1]];wo=[list(map(F,r)) for r in d['model'][2]];x=list(map(F,d['word']));rho=F(d['radius']);point=x+[F(1)]
def dot(a,b):return sum((v*w for v,w in zip(a,b)),F(0))
checks=0;labels=[]
for hist in d['histories']:
 h=[[F(0)]*4+[F(v)] for v in hist['initial_hidden']];constraints=[]
 for t in range(4):
  z=[[sum(wh[i][j]*h[j][q] for j in range(2))+wi[i]*int(q==t) for q in range(5)] for i in range(2)]
  h=[]
  for row in z:
   active=dot(row,point)>0
   if any(row):constraints.append(row if active else [-v for v in row])
   h.append(row if active else [F(0)]*5)
 logits=[[sum(w[j]*h[j][q] for j in range(2)) for q in range(5)] for w in wo]
 label=max(range(10),key=lambda j:dot(logits[j],point));assert label==hist['label'];labels.append(label)
 for k in range(10):
  if k==label:continue
  margin=[v-w for v,w in zip(logits[label],logits[k])]
  if any(margin):constraints.append(margin)
  else:assert k>label
 assert constraints==[list(map(F,c['normal'])) for c in hist['constraints']]
 for row,c in zip(constraints,hist['constraints']):
  slack=dot(row,point);assert slack==F(c['slack']) and slack>0
  assert slack-rho*sum(map(abs,row[:-1]))>=slack/2
  checks+=1
 for signs in product((-1,1),repeat=4):
  v=list(map(F,hist['initial_hidden']))
  for t in range(4):v=[max(F(0),wi[i]*(x[t]+rho*signs[t])+dot(wh[i],v)) for i in range(2)]
  assert max(range(10),key=lambda j:dot(wo[j],v))==label
assert labels[0]!=labels[1]
R=max(map(abs,x))+rho;K=d['density_integer_K'];assert F(K)>=R*R/2
lower=(2*rho/F(3)**(K+1))**4;assert lower==F(d['exact_gaussian_mass_lower_bound'])
out={'labels':labels,'exact_affine_constraints_rebuilt':checks,'direct_vertex_recurrences':32,'radius':str(rho),'strict_density_lower_bound':str(lower),'lower_float':float(lower),'cpu_seconds':time.process_time()-start,'scope':'arbitrary two fixed initial hidden states; actual prefix reachability not checked'}
Path(__file__).with_name('complement_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
