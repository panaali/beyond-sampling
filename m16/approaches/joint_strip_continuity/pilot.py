"""Single frozen trained coefficient evaluation; no model input evaluation."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,importlib.util
import torch
from scipy.special import ndtr
from strips import certify,serial
HERE=Path(__file__).parent

def main():
    stamp=time.process_time();prior=json.loads((HERE/'previous_pilot.json').read_text());path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth');sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==prior['checkpoint_sha256']
    sd=torch.load(path,map_location='cpu',weights_only=True);wi=tuple(F(float(v)) for v in sd['rnn.weight_ih_l0'][:,0]);wh=tuple(tuple(F(float(v)) for v in row) for row in sd['rnn.weight_hh_l0']);wo=tuple(tuple(F(float(v)) for v in row) for row in sd['linear.weight']);model=(wi,wh,wo);assert len(wo)==4
    spec=importlib.util.spec_from_file_location('previous',HERE/'previous_certificate.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);baseline=old.certify(model,2);assert baseline['G']==F(prior['certificate']['G'])
    proof=certify(model,2);L=float(ndtr(1.)-ndtr(-6.));U=float(ndtr(-1.)-ndtr(-6.));display=None;rigorous=None
    if proof['sufficient']:
        K=proof['K_unique'];display=dict(L=L,U=U,joint_rank_masses=[L*L,2*L*U,U*U],literal_K=float(proof['K_literal']),unique_K=float(K),conditional_need=[float(K)/z for z in (L*L,2*L*U,U*U)],separate_offdiagonal_side=float(K)/(L*U),old_uniform_y=float(baseline['G'])/min(L,U),floating_CDF_not_outward=True)
        display['old_to_new_need1_factor']=display['old_uniform_y']/display['conditional_need'][1] if K else None
        rigorous=dict(elementary_rank_lower_bounds=['1/3','1/27'],conditional_need=[9*K,F(81,2)*K,729*K],rank_bound_proof='L>2phi(1)>1/3; U>phi(2)>1/27 using sqrt(2pi)<3 and e<3; hence conditional need1 denominator2LU>2/81.')
    out=dict(checkpoint_sha256=sha,proof=proof,old_G=baseline['G'],display=display,elementary_certified_bounds=rigorous,cpu_seconds=time.process_time()-stamp,forward_evaluations=0,scope='Real-arithmetic hidden infinity-norm continuity coefficient; no interpolation accuracy or estimator-cost result.')
    (HERE/'pilot.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(json.dumps(serial(dict(sufficient=proof['sufficient'],emitted=proof['emitted'],unique_known=proof['unique_known'],unknowns=len(proof['unknown']),display=display,rigorous_bounds_float=[float(x) for x in rigorous['conditional_need']] if rigorous else None,cpu_seconds=out['cpu_seconds']))))
if __name__=='__main__':main()
