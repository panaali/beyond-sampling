"""Post-search LP path controls with analytic positive and no-crossing cases."""
import json,time
from pathlib import Path
import numpy as np
import search
from certify import matrix,exact_forward
from fractions import Fraction as F
start=time.process_time();x=np.array([3.,2.,1.,.25]);top=[0,1,2];order=[0,1,2,3];Wi=np.ones((2,1));Wh=np.diag([1.,0.]);Wo=np.array([[1.,-13.],[0.,0.],[-1.,0.],[-1.,0.]])
G,L,g=search.maps(x,Wi,Wh,Wo);good=search.propose(x,top,order,G,L,1);bad=search.propose(x,top,order,G,L,1,0);assert good['success'] and bad['success'] and min(good['t'],bad['t'])>0
winners=[]
for r in (good,bad):
 p=[F(float(v)) for v in r['x']];out,pre,g2=exact_forward(p,matrix(Wi),matrix(Wh),matrix(Wo));assert g2==g;winners.append(max(range(4),key=lambda j:out[j]))
assert winners==[1,0]
Wh=np.eye(2);Wo=np.array([[1.,0.],[0.,1.],[0.,0.],[0.,0.]]);G,L,g=search.maps(x,Wi,Wh,Wo);null=search.propose(x,top,order,G,L,1);assert not null['success'] and null['status']=='identical_output_margin'
out=dict(constructive_correct=good,constructive_wrong=bad,exact_winners=winners,tied_truth_rejected=null,lp_calls=search.CALLS,float_anchor_evaluations=2,exact_input_evaluations=2,cpu_seconds=time.process_time()-start,scope='Post-search adapter controls; exact analytic controls preceded target search.')
Path(__file__).with_name('lp_control_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('exact_winners','lp_calls','cpu_seconds')}))
