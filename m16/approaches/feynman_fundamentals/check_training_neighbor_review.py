"""Preselected independent DD/high-precision area check, before reading truth row."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path
START=time.process_time()
import numpy as np
import mpmath as mp
import torch

BASE=Path('/workspace/work/recover-small_h2_56/m16/approaches')
checkpoint=BASE/'training_axis_ladder/data/H4_T3_n2p29_seed4.pth'
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
keys=('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')
weights=tuple(state[k].numpy().astype(float).ravel() if i==0 else state[k].numpy().astype(float)
              for i,k in enumerate(keys))
rng=np.random.default_rng(23)
for copy in range(256):
    model=tuple(w+.05*rng.standard_normal(w.shape) for w in weights)
spec=importlib.util.spec_from_file_location('review_native_dd',BASE/'compiled_geometry/geometry_native.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
geometry=native.geometry(model)
mp.mp.dps=60
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def mass(triangles):
    total=mp.mpf(0)
    for triangle in triangles:
        a,b,c=[[mp.mpf(float(x)) for x in row] for row in triangle]
        ra,rb,rc=[mp.sqrt(dot(row,row)) for row in (a,b,c)]
        total+=2*mp.atan2(abs(dot(a,cross(b,c))),ra*rb*rc+dot(a,b)*rc+dot(b,c)*ra+dot(c,a)*rb)
    return total/(4*mp.pi)
good,bad=mass(geometry['good']),mass(geometry['bad'])
out=dict(training_seed=4,noise_seed=23,copy=255,power=29,
         checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
         model_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(w).tobytes() for w in model)).hexdigest(),
         accuracy=str(good),incorrect=str(bad),partition=str(good+bad),
         counts=geometry['counts'].tolist(),cpu_seconds_including_imports=time.process_time()-START,
         scope='Independent DD geometry plus60-digit spherical areas on stored floating rays; floating topology not enclosed.')
Path(__file__).with_name('training_neighbor_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
