"""One fixed exact shared-face check and unconditional event bound; no forward."""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import torch
from scipy.integrate import quad
from rational_bounds import event_bounds,serial,accounting

HERE=Path(__file__).resolve().parent
OUT=HERE/'application.json'
assert not OUT.exists(),'Preserve prior application'
assert (HERE/'controls.json').exists(),'Primitive controls required first'
sys.set_int_max_str_digits(0)
t0=time.process_time()
pins=json.loads((HERE/'manifest.json').read_text())
for path,sha in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
base=HERE.parent
prior=json.loads((base/'prefix_face_projection/pilot.json').read_text())
forms=json.loads((base/'joint_strip_continuity/pilot.json').read_text())['proof']['forms']
checkpoint=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
assert pins[str(checkpoint)]==prior['checkpoint_sha256']
sd=torch.load(checkpoint,map_location='cpu',weights_only=True)
wi=[F(float(x)) for x in sd['rnn.weight_ih_l0'][:,0]]
wh=[[F(float(x)) for x in row] for row in sd['rnn.weight_hh_l0']]
wo=[[F(float(x)) for x in row] for row in sd['linear.weight']]
assert len(wi)==2 and len(wh)==2 and len(wo)==4
assert wi[0]<0<wi[1] and wh[0][0]<0 and wh[1][0]<0
assert prior['axis']==0 and F(prior['target'])==1 and not prior['unknown_forms']
assert len(forms)==prior['source_forms']==176
moving={i for i,f in enumerate(forms) if F(f['row'][2])!=0}
zero={i for i,f in enumerate(forms) if F(f['row'][2])==0}
assert zero==set(prior['zero_state_forms']) and len(zero)==77
expected={(i,side) for i in moving for side in ['01','10']}
seen=[(r['source_index'],r['rank_side']) for r in prior['records']]
assert set(seen)==expected and len(seen)==len(expected)==198
gate=(wh[1][0],wi[1],F(0),F(0))
kappa=-wh[1][0]/wi[1]
k=kappa*abs(wi[0])
evidence=[]
for idx,r in enumerate(prior['records']):
    if not r['projection']['feasible']:continue
    f=forms[r['source_index']]
    assert f['first']==[False,True] and r['rank_side']=='10'
    matches=[]
    for j,row in enumerate(r['original_constraints']):
        row=tuple(map(F,row));scale=row[0]/gate[0]
        if scale>0 and row==tuple(scale*x for x in gate):matches.append({'constraint_index':j,'scale':scale,'row':row})
    assert matches,('missing exact gate constraint',idx)
    evidence.append({'record_index':idx,'source_index':r['source_index'],'kind':r['kind'],
                     'rank_side':r['rank_side'],'first_mask':f['first'],'matches':matches})
assert len(evidence)==14
bound=event_bounds(k)
kf=float(k);rt=math.sqrt(2);phi0=1/math.sqrt(2*math.pi)
def integrand(v):
    p=(1+math.erf(v/rt))/2
    return .25*phi0*math.exp(-v*v/2)*math.erf(v/(kf*rt))*(p*p-.25)
value,error=quad(integrand,0,math.inf,epsabs=1e-16,epsrel=2e-12,limit=100)
assert float(bound['event_probability'].lo)<=value<=float(bound['event_probability'].hi)
out={'complete':True,'source_pins':pins,'weights':{'Wi':wi,'Wh':wh,'Wo':wo},'kappa':kappa,'k':k,
     'checked_complete_occurrence_rank_pairs':198,'reused_infeasible_projection_certificates':184,
     'shared_gate_constraint':gate,'feasible_face_evidence':evidence,'bounds':bound,
     'event_scalar_integral':value,'quad_error_estimate_uncertified':error,
     'sector_probability':F(1,12),'rank10_sector_probability':F(1,24),
     'accounting':accounting(),'cpu_seconds':time.process_time()-t0,
     'peak_rss_kib_including_imports':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
     'checkpoint_loads':1,'network_forward_MC_calls':0,'scalar_reference_integrals':1,
     'application_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'scope':'Unconditional joint disagreement upper on fixed H2T4 negative-prefix/positive-target sector, conditional on inherited exhaustive exact face proof. Not actual error, constant-limit value, full accuracy or production cost.'}
with OUT.open('x') as f:json.dump(serial(out),f,indent=2);f.write('\n')
print(json.dumps({'k':kf,'kappa':float(kappa),'event_bracket':[float(bound['event_probability'].lo),float(bound['event_probability'].hi)],
    'event_scalar_integral':value,'feasible_shared_faces':len(evidence),'cpu_seconds':out['cpu_seconds'],
    'peak_rss_kib':out['peak_rss_kib_including_imports'],'exact_bit_peaks':accounting()},indent=2))
