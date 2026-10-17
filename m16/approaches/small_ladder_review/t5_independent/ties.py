"""Check review-local first-index handling on positive-volume ties."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from check import module,safe_outcomes,simplices

app=Path('/workspace/clean-neurons/m16/approaches');start=time.process_time();deadline=start+110
sys.path.insert(0,str(app/'small_h2_56'))
dd,dsha=module('ties_dd',app/'exact_ladder/cones.py')
vol,vsha=module('volume',app/'small_h2_56/volume.py');sys.modules['volume']=vol
segment,ssha=module('ties_segment',app/'small_h2_56/segment.py')
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T5_seed0.pth',weights_only=True,map_location='cpu')
base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
output=Path(__file__).with_name('tie_results.json');report=dict(sources=dict(dd=dsha,volume=vsha,segment=ssha),cases=[])
for name,wo in [('zero_logits',np.zeros_like(base[2])),('identical_nonzero_logits',np.tile(base[2][0],(5,1)))]:
    begin=time.process_time();model=(*base[:2],wo);ops=dd.Ops();counts=[0]*5;rays=[];skips=0;zeros=0
    for cell in dd.iter_cells(model,ops,counts):
        good,n=safe_outcomes(dd,cell,model,ops);skips+=n
        for r,z in good:
            new,n=simplices(dd,r,z,ops,5);rays.extend(new);zeros+=n
    rays=np.array(rays);orders,bounds=vol.plan(rays,1e-8);result=segment.grouped_mass(rays,orders,bounds,deadline)
    result.update(name=name,known_accuracy=.2,error=result['estimate']-.2,zero_determinants=zeros,tie_candidates_rejected=skips,dd_operations=ops.n,correct_simplices=len(rays),cpu_seconds=time.process_time()-begin)
    report['cases'].append(result);report['total_cpu']=time.process_time()-start;output.write_text(json.dumps(report,indent=2)+'\n');print(result,flush=True)
