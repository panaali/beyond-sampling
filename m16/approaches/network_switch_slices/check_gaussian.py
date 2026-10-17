"""Frozen same-panel natural-coordinate rule; all eight outcomes retained."""
import hashlib,json,time
from collections import Counter
import numpy as np
from .slices import Slice,ROOT
from .check_slices import HERE


def main():
    start=time.process_time()
    import torch
    previous=json.loads((HERE/'results.json').read_text())
    out=dict(cases=[],sources={str(HERE/'slices.py'):hashlib.sha256((HERE/'slices.py').read_bytes()).hexdigest()},
             quantity='Unnormalized integral on q in [.05,.95], no division by .9')
    for old in previous['cases']:
        t=time.process_time();H=old['H']
        st=torch.load(ROOT/f'work/checkpoints/H{H}_T4_seed0.pth',map_location='cpu',weights_only=True)
        model=(st['rnn.weight_ih_l0'].numpy().astype(float).ravel(),
               st['rnn.weight_hh_l0'].numpy().astype(float),st['linear.weight'].numpy().astype(float))
        s=Slice(model,old['kind'],old['configuration'])
        assert np.array_equal(s.knots,old['knots'])
        row=dict(H=H,kind=old['kind'],configuration=old['configuration'],
                 construction_cpu_seconds=time.process_time()-t,construction_work=dict(s.work),
                 saved_quantile_estimates=old['estimates'],rules=[])
        for order in (4,16,32):
            before=Counter(s.work);t=time.process_time()
            value,norm=s.integrate_gaussian(s.knots,order)
            row['rules'].append(dict(order=order,integral=value,probability_weight_sum=norm,
                                     normalization_defect=norm-.9,
                                     error_to_saved_reference=value-old['estimates']['reference_GL32'],
                                     cpu_seconds=time.process_time()-t,work=dict(s.work-before)))
        row['reference_disagreement']=abs(row['rules'][1]['integral']-row['rules'][2]['integral'])
        assert abs(row['rules'][-1]['normalization_defect'])<1e-12
        out['cases'].append(row)
        print(json.dumps({k:row[k] for k in ('H','kind','configuration','reference_disagreement')}|{'zGL4error':row['rules'][0]['error_to_saved_reference']}),flush=True)
    out['cpu_seconds']=time.process_time()-start
    (HERE/'gaussian_results.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
