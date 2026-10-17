"""Exact slope test and fixed two-probe rank-side localization."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from scipy.special import ndtr
import torch
from certificate import certify,rational_model,serial
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'middle_target_pair'))
from terminal import MiddleTarget
from interpolation import InterpolatedMiddle


def run():
    start=time.process_time();path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
    w=torch.load(path,map_location='cpu',weights_only=True)
    a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    model=a[0].ravel(),a[1],a[2]
    proof=certify(rational_model(model),2)
    prefix=np.array([-.25,.25]);states=np.maximum(prefix[:,None]*model[0],0.)
    old=InterpolatedMiddle(model,states);u=1.;q=float((ndtr(u)-ndtr(-6.))/(1-2*ndtr(-6.)))
    matrices,error=old.pieces.matrices(u,q,old.stats,old.deadline)
    grid=old.penultimate.values(u,q)
    side=np.stack([m@grid for m in matrices],axis=1)
    old.stats['sparse_value_products']+=2*sum(m.nnz for m in matrices)
    saved=json.loads((HERE.parent/'middle_target_pair/nonzero_prefix.json').read_text());rows=[]
    for i,h in enumerate(states):
        obj=MiddleTarget(model,h,u);values,info=obj.probabilities()
        exact=np.asarray(info['conditional_by_sides']);delta=side[i]-exact
        need1=(side[i,0,1]+side[i,1,0])/2
        assert abs(need1-saved['rows'][i]['old_interpolated'][1])<1e-14
        assert np.max(abs(exact-np.asarray(saved['rows'][i]['pair_info']['conditional_by_sides'])))<1e-14
        assert abs((delta[0,1]+delta[1,0])/2-saved['rows'][i]['old_minus_pair'][1])<1e-14
        rows.append(dict(prefix=float(prefix[i]),target=u,need=1,old_sides=side[i].tolist(),
                         exact_sides=exact.tolist(),side_errors=delta.tolist(),
                         need1_error_parts=[float(delta[0,1]/2),float(delta[1,0]/2)],
                         pair_work=dict(obj.stats)))
    L=float(ndtr(u)-ndtr(-6.));U=float(ndtr(-u)-ndtr(-6.))
    return dict(certificate=proof,lipschitz_coefficient_float=None if not proof['sufficient'] else float(proof['G'])/min(L,U),
                coefficient_display_not_outward=True,rows=rows,old_kernel_error_diagnostic=error,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),domain=old.domain,
                old_work=dict(old.stats),old_penultimate_work=dict(old.penultimate.stats),
                cpu_seconds=time.process_time()-start)


if __name__=='__main__':
    r=run();(HERE/'pilot.json').write_text(json.dumps(serial(r),indent=2)+'\n')
    print(json.dumps(serial(dict(sufficient=r['certificate']['sufficient'],G=r['certificate']['G'],coefficient=r['lipschitz_coefficient_float'],rows=[{k:v for k,v in row.items() if k!='pair_work'} for row in r['rows']],cpu=r['cpu_seconds']))))
