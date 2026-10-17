from collections import defaultdict
from fractions import Fraction as F
import hashlib,json,time
from pathlib import Path
import mpmath as mp
import torch
from certificate import rational_model,find,validate,serial,history,box_radius
HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True);a=[w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];model=rational_model((a[0].ravel(),a[1],a[2]));start=time.process_time()
r=find(model);r['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();r['source_sha256']=hashlib.sha256((HERE/'certificate.py').read_bytes()).hexdigest();r['model']=model
if r['found']:
 r['validation']=validate(model,r);mp.mp.dps=80
 def mm(q):return mp.mpf(q.numerator)/q.denominator
 rho=mm(r['radius']);prob=mp.mpf(1)
 for x in r['word']:
  x=mm(x);prob*= (mp.erf((x+rho)/mp.sqrt(2))-mp.erf((x-rho)/mp.sqrt(2)))/2
 r['gaussian_mass_80_digit']=str(prob);r['rational_lower_bound_float']=float(r['exact_gaussian_mass_lower_bound'])
 assert prob>mm(r['exact_gaussian_mass_lower_bound'])>0
 scale=F(1,2**30);scaled=(tuple(x*scale for x in model[0]),model[1],tuple(tuple(x/scale for x in row) for row in model[2]));stats=defaultdict(int)
 scaledhist=[history(scaled,tuple(x*scale for x in h['initial_hidden']),r['word'],stats) for h in r['histories']]
 assert [h['label'] for h in scaledhist]==[h['label'] for h in r['histories']]
 assert box_radius(scaledhist)==r['radius']
 sr=dict(r);sr['histories']=scaledhist;r['scale_control']=validate(scaled,sr);r['scale_control']['factor']=scale;r['scale_work']=dict(stats)
r['total_cpu_seconds']=time.process_time()-start
(HERE/'pilot.json').write_text(json.dumps(serial(r),indent=2)+'\n')
print(json.dumps(serial({k:r[k] for k in ('found','word','radius','rational_lower_bound_float','gaussian_mass_80_digit','stats','total_cpu_seconds') if k in r})))
