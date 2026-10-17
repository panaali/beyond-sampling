import json
from pathlib import Path
import numpy as np
import torch
import estimator
import tail
original=tail.tail_mass
output=Path(__file__).with_name('moment_failure.json')
def captured(vertices,k):
    try:return original(vertices,k)
    except ArithmeticError as e:
        output.write_text(json.dumps(dict(vertices=np.asarray(vertices).tolist(),prefix_dimension=k,error=str(e)),indent=2)+'\n')
        raise

tail.tail_mass=captured
w=torch.load('/workspace/clean-neurons/work/checkpoints/H4_T4_seed0.pth',weights_only=True,map_location='cpu')
base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
rng=np.random.default_rng(51001);m=tuple(w+.0025*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
estimator.angular_estimate(m,4)
