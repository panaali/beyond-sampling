"""Check corrected proposal against retained matrix and all noisy DD decisions."""
import json,time
from pathlib import Path
import numpy as np
import torch
from ..accounting import Audit
from .bounded_rank import bounded_rank,AmbiguousRank
from .geometry import enumerate_correct
HERE=Path(__file__).resolve().parent;start=time.process_time()
failure=json.loads((HERE/'rank_failure.json').read_text());a=Audit()
rank=bounded_rank(np.array(failure['matrix']),a);assert rank==3
try:bounded_rank(np.diag([1.,1.,1.,1.,1e-9]),Audit())
except AmbiguousRank:pass
else:raise AssertionError('threshold must remain ambiguous')
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T5_seed0.pth',weights_only=True,map_location='cpu');base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
rng=np.random.default_rng(1);model=tuple(x+rng.normal(size=x.shape)*np.sqrt(np.mean(x*x))*.02 for x in base)
audit=Audit();rays,stats,_=enumerate_correct(model,audit,validate_rank=True)
out=dict(captured_matrix_rank=rank,captured_matrix_cost=a.report(),threshold_ambiguity_retained=True,geometry=stats,correct_simplices=len(rays),geometry_cost=audit.report(),cpu_seconds=time.process_time()-start)
(HERE/'rank_correction.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
