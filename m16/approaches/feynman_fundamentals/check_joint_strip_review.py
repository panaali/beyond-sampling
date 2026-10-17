"""Matrix-free forced recurrence coefficient and dedup check."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from fractions import Fraction as F
from pathlib import Path
import importlib.util,json,time
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_78/m16/approaches/joint_strip_continuity/strips.py');spec=importlib.util.spec_from_file_location('strip_author',src);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
wi=(F(2,3),F(-5,7));wh=((F(3,2),F(-2,5)),(F(7,3),F(4,9)));wo=((F(2),F(-1)),(F(0),F(1)),(F(-3),F(2)),(F(1),F(3)));model=(wi,wh,wo);proof=a.certify(model,2);checked=0
for form in proof['forms']:
 for axis in range(5):
  coordinates=[F(int(j==axis)) for j in range(5)];x,y,h0,h1,u=coordinates;h=[h0,h1]
  for inp,gates in [(x,form['first']),(u,form['middle'])]:
   z=[wi[k]*inp+sum(wh[k][j]*h[j] for j in range(2)) for k in range(2)];h=[z[k]*gates[k] for k in range(2)]
  z=[wi[k]*y+sum(wh[k][j]*h[j] for j in range(2)) for k in range(2)]
  expected=z[form['neuron']] if form['kind']=='final_gate' else sum((wo[2][k]-wo[form['competitor']][k])*form['final'][k]*z[k] for k in range(2))
  assert expected==form['row'][axis];checked+=1
unique={};literal=F(0)
for form in proof['forms']:
 row=form['row'];state=abs(row[2])+abs(row[3]);normal=max(abs(row[0]),abs(row[1]));
 if not state or not normal:continue
 ratio=state/normal;literal+=ratio;k=a.key(row)
 if k in unique:assert unique[k]==ratio
 unique[k]=ratio
assert literal==proof['partial_literal'] and sum(unique.values(),F(0))==proof['partial_unique']
# Independently reconstruct every trained saved coefficient with scalar recurrences.
import torch
from scipy.special import ndtr
pilotpath=src.parent/'pilot.json';pilot=json.loads(pilotpath.read_text());sd=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',weights_only=True,map_location='cpu');wi=tuple(F(float(v)) for v in sd['rnn.weight_ih_l0'][:,0]);wh=tuple(tuple(F(float(v)) for v in r) for r in sd['rnn.weight_hh_l0']);wo=tuple(tuple(F(float(v)) for v in r) for r in sd['linear.weight']);trained=0;literal=F(0);unique={};yunique={}
for form in pilot['proof']['forms']:
 row=tuple(map(F,form['row']))
 for axis in range(5):
  x,y,h0,h1,u=[F(int(j==axis)) for j in range(5)];h=[h0,h1]
  for inp,gates in [(x,form['first']),(u,form['middle'])]:
   z=[wi[k]*inp+sum(wh[k][j]*h[j] for j in range(2)) for k in range(2)];h=[z[k]*gates[k] for k in range(2)]
  z=[wi[k]*y+sum(wh[k][j]*h[j] for j in range(2)) for k in range(2)]
  expected=z[form['neuron']] if form['kind']=='final_gate' else sum((wo[2][k]-wo[form['competitor']][k])*form['final'][k]*z[k] for k in range(2));assert expected==row[axis];trained+=1
 state=abs(row[2])+abs(row[3])
 if state:
  normal=max(abs(row[0]),abs(row[1]));assert normal;ratio=state/normal;literal+=ratio;key=tuple(v/next(z for z in row if z) for v in row);unique[key]=ratio;assert row[1];yunique[key]=state/abs(row[1])
K=sum(unique.values(),F(0));assert K==F(pilot['proof']['K_unique']) and literal==F(pilot['proof']['K_literal']);assert sum(yunique.values(),F(0))==F(pilot['matched_y_only_ablation']['y_only_unique']);assert F(pilot['elementary_certified_bounds']['conditional_need'][1])==F(81,2)*K
L=ndtr(1)-ndtr(-6);U=ndtr(-1)-ndtr(-6);conditional=float(K)/(2*L*U);assert abs(conditional-pilot['display']['conditional_need'][1])<1e-10
out=dict(trained_exact_coefficient_checks=trained,trained_unique_forms=len(unique),trained_K=float(K),trained_need1_display=conditional,trained_elementary_need1=float(F(81,2)*K),noncommuting_exact_coefficient_checks=checked,forms=len(proof['forms']),dedup_exact_ratio_checks=True,review_cpu=time.process_time()-start,scope='Independent matrix-free forced recurrence source control; theorem and rank normalization separately reviewed, no trained forward evaluation or interpolation claim.')
(here/'joint_strip_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
