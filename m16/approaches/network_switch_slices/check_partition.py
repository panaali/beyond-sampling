import json,time
import numpy as np
from .slices import ROOT
from .check_slices import HERE
from .contract import contract


def main():
    start=time.process_time()
    import torch
    out=dict(controls=[],cases=[])
    def save():
        out['cpu_seconds']=time.process_time()-start
        (HERE/'partition_results.json').write_text(json.dumps(out,indent=2)+'\n')
    def model(H):
        st=torch.load(ROOT/f'work/checkpoints/H{H}_T4_seed0.pth',map_location='cpu',weights_only=True)
        return (st['rnn.weight_ih_l0'].numpy().astype(float).ravel(),st['rnn.weight_hh_l0'].numpy().astype(float),st['linear.weight'].numpy().astype(float))
    m=model(2);constant=(m[0],m[1],np.zeros_like(m[2]))
    for order in (3,5):
        row=contract(constant,order,16,0,deadline=start+105)
        row['constant_accuracy_defect']=row['normalized_estimate']-.25
        row['raw_identity_defect']=row['estimate']-(1-.75*row['partition_mass'])
        out['controls'].append(row);save()
        assert abs(row['constant_accuracy_defect'])<1e-12 and abs(row['raw_identity_defect'])<1e-12
    truths={2:.30773074302315223,4:.47701312599404416}
    for H in (2,4):
        for order in (3,5):
            row=contract(model(H),order,16,0,deadline=start+105)
            row.update(H=H,T=4,reference=truths[H],raw_error=row['estimate']-truths[H],
                       normalized_error=row['normalized_estimate']-truths[H])
            out['cases'].append(row);save()
            print(json.dumps({k:row[k] for k in ('H','outer_order','partition_mass','raw_error','normalized_error','total_cpu_seconds')}),flush=True)
    out['complete']=True;save()


if __name__=='__main__':main()
