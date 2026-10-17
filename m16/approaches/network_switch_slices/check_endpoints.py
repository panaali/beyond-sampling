"""Extend the frozen slices with an explicit omitted endpoint mass bound."""
import json,time
from collections import Counter
import numpy as np
from .slices import Slice,ROOT,walker
from .check_slices import HERE,reference_values


def main():
    start=time.process_time()
    import torch
    old=json.loads((HERE/'results.json').read_text())
    delta=1e-6;_,z23,z12=walker.tables(4)
    out=dict(delta=delta,slice_omitted_mass_bound=2*delta,
             accuracy_omitted_mass_bound=2*delta*(z23+z12),cases=[])
    def save():
        out['cpu_seconds']=time.process_time()-start
        (HERE/'endpoint_results.json').write_text(json.dumps(out,indent=2)+'\n')
    q=np.array([2e-6,1e-5,1e-4,.001,.01,.05,.2,.35,.5,.65,.8,.95,.99,.999,.9999,.99999,1-2e-6])
    for previous in old['cases']:
        H=previous['H'];t=time.process_time()
        st=torch.load(ROOT/f'work/checkpoints/H{H}_T4_seed0.pth',map_location='cpu',weights_only=True)
        model=(st['rnn.weight_ih_l0'].numpy().astype(float).ravel(),st['rnn.weight_hh_l0'].numpy().astype(float),st['linear.weight'].numpy().astype(float))
        s=Slice(model,previous['kind'],previous['configuration'],qlo=delta,qhi=1-delta)
        row=dict(H=H,kind=previous['kind'],configuration=previous['configuration'],
                 construction_cpu_seconds=time.process_time()-t,construction_work=dict(s.work),rules=[])
        out['cases'].append(row)
        value=s.values(q);ref,cost=reference_values(model,s.kind,previous['configuration'],q)
        row.update(pointwise_defect=float(np.max(abs(value-ref))),pointwise_q=q.tolist(),
                   pointwise_values=value.tolist(),pointwise_reference=ref.tolist(),reference_work=cost)
        save();assert row['pointwise_defect']<1e-8,row
        for order in (4,16,32):
            before=Counter(s.work);t=time.process_time();v,norm=s.integrate_gaussian(s.knots,order)
            row['rules'].append(dict(order=order,value=v,normalization_defect=norm-(1-2*delta),
                                     cpu_seconds=time.process_time()-t,work=dict(s.work-before)))
        parts=[]
        for lo,hi in [(delta,.05),(.05,.95),(.95,1-delta)]:
            knots=np.unique(np.r_[lo,s.knots[(s.knots>lo)&(s.knots<hi)],hi])
            value,norm=s.integrate_gaussian(knots,32);parts.append(value)
        row['partition_parts']=parts
        row['partition_defect']=sum(parts)-row['rules'][-1]['value']
        row['interior_reference_defect']=parts[1]-previous['estimates']['reference_GL32']
        row['reference_disagreement']=abs(row['rules'][1]['value']-row['rules'][2]['value'])
        save()
        assert abs(row['rules'][-1]['normalization_defect'])<1e-12
        assert row['reference_disagreement']<1e-8
        assert min(parts)>-1e-12 and abs(row['partition_defect'])<1e-10
        print(json.dumps({k:row[k] for k in ('H','kind','configuration','pointwise_defect','reference_disagreement','partition_defect')}),flush=True)
    out['passed']=True;save()


if __name__=='__main__':main()
