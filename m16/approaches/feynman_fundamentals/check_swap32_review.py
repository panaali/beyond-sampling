import sys,time,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/work/recover-small_h2_78/m16/approaches/stratum_transport');sys.path.insert(0,str(ROOT))
from event_volume import EventInterfaceOperator
start=time.process_time();w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth',map_location='cpu',weights_only=True)
m=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));P=np.array([[0.,1.],[1.,0.]])
m=(P@m[0].ravel(),P@m[1]@P,m[2]@P)
op=EventInterfaceOperator(m,32,'loglinear',True);s=op.initial()
for t in range(4):s,out=op.step(s)
a=next(r for r in json.loads((ROOT/'split_loglinear_results.json').read_text())['rows'] if r['model']=='trained_H2T8' and r['grid']==32)
ref=next(r['reference'] for r in json.loads((ROOT/'fourth_step_results.json').read_text())['rows'] if r['model']=='trained_H2T8' and r['grid']==32)
result=dict(grid=32,coordinate_swap=True,output=out.tolist(),difference_from_original=float(np.max(abs(out-a['steps'][-1]['output_masses']))),endpoint_error=float(np.max(abs(out-ref['output_masses']))),cpu_seconds=time.process_time()-start,work=op.report())
Path(__file__).with_name('swap32_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='work'})
