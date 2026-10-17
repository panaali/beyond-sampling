"""Capture the retained ambiguity; SVD is a diagnostic only."""
import json,time
from pathlib import Path
import numpy as np
import torch
from ..accounting import Audit
from .geometry import enumerate_correct
HERE=Path(__file__).resolve().parent;start=time.process_time()
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T5_seed0.pth',weights_only=True,map_location='cpu')
base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
rng=np.random.default_rng(1);model=tuple(x+rng.normal(size=x.shape)*np.sqrt(np.mean(x*x))*.02 for x in base)
audit=Audit()
try:enumerate_correct(model,audit)
except ArithmeticError as exc:
 matrix=exc.rank_matrix;singular=np.linalg.svd(matrix,compute_uv=False)
 out=dict(error=str(exc),matrix=matrix.tolist(),singular_values=singular.tolist(),svd_rank_at_1e9=int(np.sum(singular>1e-9)),validation_svd_calls=1,partial_production_cost=audit.report(),cpu_seconds=time.process_time()-start)
 (HERE/'rank_failure.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(error=str(exc),shape=matrix.shape,singular_values=singular.tolist(),cpu=out['cpu_seconds'])))
else:raise AssertionError('expected retained ambiguity')
