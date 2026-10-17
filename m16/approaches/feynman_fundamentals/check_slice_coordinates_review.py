"""Matched-panel coordinate/topology discriminator on fixed eight slices."""
import sys,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import torch
sys.path.insert(0,'/workspace/work/recover-small_h2_78')
from m16.approaches.network_switch_slices.slices import Slice,vertical_mass,ROOT
SOURCE=Path('/workspace/work/recover-small_h2_78/m16/approaches/network_switch_slices')
start=time.process_time();old=json.loads((SOURCE/'results.json').read_text());rows=[]
x,w=np.polynomial.legendre.leggauss(4)
for case in old['cases']:
 H=case['H'];state=torch.load(ROOT/f'work/checkpoints/H{H}_T4_seed0.pth',map_location='cpu',weights_only=True)
 model=[state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];model[0]=model[0].ravel()
 s=Slice(model,case['kind'],case['configuration']);network=s.toz(s.knots)
 knots={'network':network,'equal_width_z':np.linspace(s.zlo,s.zhi,len(network)),'geometry_only':np.unique(np.r_[s.zlo,s.domain[:,0],s.zhi])}
 result=[]
 for name,ends in knots.items():
  lo,hi=ends[:-1,None],ends[1:,None];z=((lo+hi)/2+(hi-lo)*x/2).ravel();ww=(w*(hi-lo)/2).ravel()
  weight=ww*np.exp(-z*z/2)/np.sqrt(2*np.pi)/(1-s.P if s.upper else s.P)
  work=Counter();value=vertical_mass([s.domain],z,work)-vertical_mass(s.correct,z,work)
  estimate=float(weight@value)
  result.append(dict(rule=name,panels=len(ends)-1,nodes=len(z),estimate=estimate,error_to_saved=estimate-case['estimates']['reference_GL32'],normalizer=float(weight.sum()),work=dict(work)))
 rows.append(dict(H=H,kind=case['kind'],configuration=case['configuration'],rules=result))
out=dict(cases=rows,cpu_seconds=time.process_time()-start,source_sha256=hashlib.sha256((SOURCE/'slices.py').read_bytes()).hexdigest(),scope='shared polygon geometry, independent coordinate-rule assembly; same eight slices')
Path(__file__).with_name('slice_coordinates_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
