"""Independent integer-scaled recurrence on all64 frozen exact witnesses."""
from fractions import Fraction as F
from pathlib import Path
import json,time,math,resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
import numpy as np
import torch
here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_56/m16/approaches/error_cone_semantics');base=Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review');d=json.loads((src/'floating.json').read_text());v=np.load(src/'floating.npz');rays=np.load(base/'error_captured.npz')['rays'];sd=torch.load('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth',weights_only=True,map_location='cpu');W=[sd[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];den=max(F(float(x)).denominator for a in W for x in a.flat);wi=[int(F(float(x))*den) for x in W[0][:,0]];wh=[[int(F(float(x))*den) for x in row] for row in W[1]];wo=[[int(F(float(x))*den) for x in row] for row in W[2]];start=time.process_time();saved_margin=v['margin']
assert not d['failure_count'];selection=sorted(range(len(saved_margin)),key=lambda i:(abs(saved_margin[i]),i))[:64];assert selection==d['exact_selection'];assert np.array_equal(np.argmax(v['logits'],axis=1),v['prediction']);assert np.all(v['prediction']!=v['truth']);assert np.array_equal(np.argsort(-v['inputs'],axis=1,kind='stable')[:,1],v['truth']);assert np.array_equal(v['truth'],v['expected_truth'])
seen=[];margins=[]
for chunk in range(4):
 data=json.loads((src/f'exact_{chunk}.json').read_text());assert data['complete']
 for row in data['rows']:
  i=row['index'];seen.append(i);x=[sum((F(float(q)) for q in rays[i,:,j]),F(0)) for j in range(4)];assert x==list(map(F,row['exact_inputs']));xd=max(q.denominator for q in x);xi=[int(q*xd) for q in x];h=[0]*8;gates=[]
  for t,value in enumerate(xi):
   z=[wi[k]*value*den**t+sum(wh[k][j]*h[j] for j in range(8)) for k in range(8)];gates.append([(q>0)-(q<0) for q in z]);h=[max(q,0) for q in z]
  outputs=[F(sum(w*q for w,q in zip(line,h)),xd*den**5) for line in wo];assert outputs==list(map(F,row['exact_logits']));assert gates==row['gate_signs'] and all(q for g in gates for q in g);truth=sorted(range(4),key=lambda j:(-x[j],j))[1];pred=max(range(4),key=lambda j:outputs[j]);assert truth==row['expected_truth']==row['truth'] and pred==row['prediction']!=truth;assert len(set(x))==4;gap=outputs[truth]-max(q for j,q in enumerate(outputs) if j!=truth);assert gap==F(row['exact_margin'])<0;margins.append(float(gap))
assert seen==selection
out=dict(saved_float_witnesses=len(v['inputs']),independent_integer_exact_witnesses=len(seen),selected_by_smallest_abs_margin=True,exact_gate_zeros=0,exact_margin_range=[min(margins),max(margins)],review_cpu_excluding_load=time.process_time()-start,scope='Finite positive-ray-sum point semantics only; no whole-cone containment/completeness or mass certificate.')
(here/'error_semantics_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
