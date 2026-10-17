"""Frozen complete-integral DEV experiment. No test models or fitted selection."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[k]='1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from estimator import estimate_details, WALK

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
a=p.parse_args()
OUT=Path(__file__).resolve().parent
s=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
base=tuple(s[k].detach().numpy().astype(np.float64) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
base=(base[0][:,0],base[1],base[2])
reference=json.loads((OUT.parent/'sci_tao_v2/dev_benchmark.json').read_text())
records=[]; start=time.process_time()
for sigma in (.0025,.02):
    rng=np.random.default_rng(1)
    model=tuple(w+sigma*np.sqrt((w*w).mean())*rng.standard_normal(w.shape) for w in base)
    name=f'dev_s{sigma}_0'
    ref=next(r for r in reference if r['model']==name)
    totalfaces=0
    for order,rank in ((3,2),(3,4),(5,2),(5,4)):
        begin=time.process_time()
        try:
            acc,cost,detail=estimate_details(model,order,rank,sweeps=2,audit=True)
            faces=sum(d['faces'] for d in detail)
            totalfaces+=faces
            variance=ref['diagnostic_truth']*(1-ref['diagnostic_truth'])/cost
            row=dict(model=name,order=order,rank=rank,sweeps=2,accuracy=acc,
                cost_fpe=cost,faces=faces,cumulative_faces_model=totalfaces,
                backend=WALK.__module__,cpu_seconds=time.process_time()-begin,
                diagnostic_truth=ref['diagnostic_truth'],truth_se=ref['truth_se'],truth_n=ref['truth_n'],
                error_pp=100*(acc-ref['diagnostic_truth']),
                raw_single_error_squared_over_sampling_variance=(acc-ref['diagnostic_truth'])**2/variance,
                noise_corrected_single_ratio=((acc-ref['diagnostic_truth'])**2-ref['truth_se']**2)/variance,
                detail=detail)
        except Exception as exc:
            row=dict(model=name,order=order,rank=rank,failure=repr(exc),cpu_seconds=time.process_time()-begin)
        records.append(row)
        (OUT/'dev_results.json').write_text(json.dumps(dict(records=records,cpu_seconds=time.process_time()-start),indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='detail'}),flush=True)
        if totalfaces>32768 or time.process_time()-start>290: raise RuntimeError('initial authorized limit reached')
