"""Frozen two-case factorial; all references are scoring-only."""
import hashlib,json,time,traceback
from pathlib import Path
import numpy as np
from engine import calculate,OLD,HERE

def run():
    start=time.process_time()
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
    state=torch.load(path,map_location='cpu',weights_only=True)
    base=tuple(state[k].detach().numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
    base=(base[0][:,0],base[1],base[2]);rng=np.random.default_rng(47021)
    noisy=tuple(w+.02*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base)
    reference={r['name']:r for r in json.loads((OLD/'pilot.json').read_text())['rows'] if not r['shuffled']}
    out=dict(rows=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),preparation_cpu=time.process_time()-start)
    for name,model in (('T4_copy0',base),('T4_rms0.02_copy0',noisy)):
        prior=reference[name]
        for angular,radial in ((2,2),(2,4),(4,2),(4,4)):
            row=dict(name=name,angular=angular,radial=radial,reference=prior['reference'],reference_bound=prior['reference_bound'],previous_mean=prior['estimate'])
            try:
                value,stats=calculate(model,angular,radial,16,seconds=100-(time.process_time()-start))
                fine,finestats=calculate(model,angular,radial,32,seconds=100-(time.process_time()-start))
                row.update(estimate=value,error=value-prior['reference'],diagnostic=fine,diagnostic_difference=fine-value,stats=stats,diagnostic_stats=finestats)
            except Exception as exc:row.update(failure=repr(exc),traceback=traceback.format_exc())
            out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
            (HERE/'pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
            print({k:v for k,v in row.items() if k not in ('stats','diagnostic_stats','traceback')},flush=True)
            if 'failure' in row:return
            if out['total_cpu_seconds']>100:raise RuntimeError('bounded pilot CPU')
    print('complete CPU',out['total_cpu_seconds'],flush=True)

if __name__=='__main__':run()
