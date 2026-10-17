"""Independent exact direct recurrence and common-neighborhood verification."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
import torch
here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_78/m16/approaches/gate_symbol_witness');d=json.loads((src/'certificate.json').read_text());p=Path('/workspace/work/t39_sources/example_checkpoint.pth');assert hashlib.sha256(p.read_bytes()).hexdigest()==d['model_sha256'];sd=torch.load(p,weights_only=True,map_location='cpu');wi=[F(float(x)) for x in sd['rnn.weight_ih_l0'][:,0]];wh=[[F(float(x)) for x in row] for row in sd['rnn.weight_hh_l0']];wo=[[F(float(x)) for x in row] for row in sd['linear.weight']];start=time.process_time();c=d['certificate'];radius=F(c['common_radius']);assert radius>0;records=[]
for point in c['pair']:
 x=list(map(F,point['x']));h=[F(0)]*16;lip=h.copy();masks=[];minratio=None
 for value in x:
  z=[wi[i]*value+sum(wh[i][j]*h[j] for j in range(16)) for i in range(16)];lip=[abs(wi[i])+sum(abs(wh[i][j])*lip[j] for j in range(16)) for i in range(16)];assert all(z);assert all(4*radius*l<=abs(v) for v,l in zip(z,lip));masks.append([v>0 for v in z]);h=[max(v,0) for v in z]
 out=[sum(a*b for a,b in zip(w,h)) for w in wo];assert out==list(map(F,point['outputs']));assert masks==c['gates'];winner=max(range(10),key=lambda j:out[j]);assert winner==point['winner'];order=sorted(range(10),key=lambda j:x[j],reverse=True);assert order==c['rank_order'] and order[:3]==c['top'];assert all(x[a]-x[b]>=8*radius for a,b in zip(order,order[1:]))
 for j in range(10):
  if j==winner:continue
  bound=sum(abs(wo[winner][i]-wo[j][i])*lip[i] for i in range(16));assert out[winner]-out[j]>=4*radius*bound and out[winner]>out[j]
 assert point['correct']==(winner==order[1]);records.append(dict(winner=winner,correct=point['correct'],top_values=[str(x[i]) for i in c['top']],gate_constraints=len(masks)*16))
assert records[0]['top_values']==records[1]['top_values'] and records[0]['correct']!=records[1]['correct']
margin=json.loads((src/'margin_results.json').read_text());adj=[a-b for a,b in zip(wo[3],wo[6])];coeff=[F(0)]*10
for t in range(9,-1,-1):
 masked=[v*int(g) for v,g in zip(adj,c['gates'][t])];coeff[t]=sum(v*w for v,w in zip(masked,wi));adj=[sum(masked[i]*wh[i][j] for i in range(16)) for j in range(16)]
assert coeff==list(map(F,margin['margin_coefficients']))
x0=list(map(F,c['pair'][0]['x']));x1=list(map(F,c['pair'][1]['x']));m0=sum(a*b for a,b in zip(coeff,x0));m1=sum(a*b for a,b in zip(coeff,x1));lam=m0/(m0-m1);assert lam==F(margin['crossing_parameter']) and 0<lam<1
assert all(coeff[j] for j in range(10) if j not in c['top']);star=[v+lam*(w-v) for v,w in zip(x0,x1)];h=[F(0)]*16
for t,value in enumerate(star):
 z=[wi[i]*value+sum(wh[i][j]*h[j] for j in range(16)) for i in range(16)];assert all(z) and [v>0 for v in z]==c['gates'][t];h=[max(v,0) for v in z]
outputs=[sum(a*b for a,b in zip(w,h)) for w in wo];assert outputs[3]==outputs[6];other=min(outputs[3]-v for j,v in enumerate(outputs) if j not in (3,6));assert other==F(margin['other_output_min_margin'])>0
out=dict(reverse_adjoint_coefficients_match=True,crossing_parameter=float(lam),other_output_min_margin=float(other),points=records,common_radius=str(radius),common_radius_float=float(radius),review_cpu_excluding_torch_load=time.process_time()-start,scope='Exact binary-weight recurrence and common open input boxes; no practical probability/accuracy floor or impossibility for weighted lower-input summaries.')
(here/'gate_symbol_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
