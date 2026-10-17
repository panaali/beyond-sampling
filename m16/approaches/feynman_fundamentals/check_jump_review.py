import sys,math,time,json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.special import ndtr
ROOT=Path('/workspace/work/recover-small_h2_78/m16/approaches/stratum_transport')
sys.path.insert(0,str(ROOT))
from two_input import Law
start=time.process_time()
import torch
data=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth',map_location='cpu',weights_only=True)
model=tuple(data[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=(model[0].ravel(),*model[1:])
wi,wh,wo=model;law=Law(model);L=6.;phi0=1/math.sqrt(2*math.pi);g={'axis0':0.,'axis1':0.};cells=[]
for cell in law.cells:
 if cell['kind'] not in g:continue
 a=cell['B'][int(cell['kind'][-1])];v=[-a[1],a[0]];norm=math.sqrt(float(a[0]**2+a[1]**2));lo,hi=-math.inf,math.inf;valid=True
 constraints=[([F(1),F(0)],0),([F(-1),F(0)],-L),([F(0),F(1)],-L),([F(0),F(-1)],-L)]
 if cell['lo'] is not None:constraints.append(([-cell['lo'],F(1)],0))
 if cell['hi'] is not None:constraints.append(([cell['hi'],F(-1)],0))
 for n,b in constraints:
  coefficient=n[0]*v[0]+n[1]*v[1]
  if coefficient:
   root=b*norm/float(coefficient)
   if coefficient>0:lo=max(lo,root)
   else:hi=min(hi,root)
  elif b==0 and n[0]*a[0]+n[1]*a[1]<0:valid=False
 p=phi0/norm*max(0.,float(ndtr(hi)-ndtr(lo))) if valid else 0.
 g[cell['kind']]+=p;cells.append(dict(kind=cell['kind'],limit=p,valid=valid,interval=[lo,hi]))
# Strictly positive input and nondegenerate recurrent ray: only first-step reset regenerates the two-step input ray.
g['input_ray']=(float(ndtr(L))-.5)*phi0
entries=[];coefficient=0.
for kind,d in [('axis0',np.array([1.,0.])),('axis1',np.array([0.,1.])),('input_ray',wi)]:
 v=wh@d;det=float(v[0]*wi[1]-v[1]*wi[0]);signed=-g[kind]/det;coefficient+=signed
 entries.append(dict(kind=kind,density_at_zero=g[kind],det_Wd_w=det,signed_jump_coefficient=signed))
out=dict(entries=entries,cells=cells,total_jump_coefficient=coefficient,jump_at_h_equals_w=coefficient*phi0*math.exp(-.5),cpu_seconds=time.process_time()-start,weights=[x.tolist() for x in model])
Path(__file__).with_name('jump_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
