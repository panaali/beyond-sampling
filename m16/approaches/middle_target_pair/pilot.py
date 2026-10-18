"""Frozen nine-query penultimate representation discriminator."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from terminal import MiddleTarget
from interpolation import InterpolatedMiddle
HERE=Path(__file__).resolve().parent


def run():
    start=time.process_time();path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
    w=torch.load(path,map_location='cpu',weights_only=True)
    arrays=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    model=arrays[0].ravel(),arrays[1],arrays[2]
    prefixes=(-1.,0.,1.);states=np.maximum(np.array(prefixes)[:,None]*model[0],0.)
    old=InterpolatedMiddle(model,states);rows=[];axis=0.
    for u in (-1.,0.,1.):
        approx,error=old.values(u)
        for index,(x0,h) in enumerate(zip(prefixes,states)):
            obj=MiddleTarget(model,h,u);value,info=obj.probabilities();swap,_=obj.probabilities(swap=True)
            axis=max(axis,float(np.max(abs(value-swap))))
            need=None if x0==u else int(x0<u)
            delta=approx[index]-value
            rows.append(dict(prefix=x0,hidden=h.tolist(),target=u,exact_pair=value.tolist(),
                             old_interpolated=approx[index].tolist(),old_minus_pair=delta.tolist(),
                             rank_consistent_need=need,input_tie=x0==u,
                             rank_consistent_error=None if need is None else float(delta[need]),
                             pair_info=info,pair_work=dict(obj.stats),old_kernel_error_diagnostic=error))
    assert axis<1e-10
    return dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('terminal.py','interpolation.py','pilot.py')},
                rows=rows,axes_max=axis,domain=old.domain,old_work=dict(old.stats),
                old_penultimate_work=dict(old.penultimate.stats),
                max_local_error=float(max(np.max(np.abs(r['old_minus_pair'])) for r in rows)),
                max_rank_consistent_error=max(abs(r['rank_consistent_error']) for r in rows if r['rank_consistent_error'] is not None),
                rank_consistent_queries=sum(r['rank_consistent_need'] is not None for r in rows),
                cpu_seconds=time.process_time()-start,population_weighted_error=False)


if __name__=='__main__':
    r=run();(HERE/'pilot.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('axes_max','max_local_error','max_rank_consistent_error','rank_consistent_queries','cpu_seconds')}))
