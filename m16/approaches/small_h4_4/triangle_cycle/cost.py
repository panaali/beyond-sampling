"""Frozen matched-process CPU comparison; counts are not a portable FLOP ledger."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from run import integrate, load_geometry, adaptive_mass, HERE

start = time.process_time()
sys.path.insert(0, '/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals')
from triangle_batch import signed_l1_triangles
import torch

weights = torch.load('/workspace/clean-neurons/work/checkpoints/H4_T4_seed0.pth',
                     weights_only=True, map_location='cpu')
base = tuple(weights[k].numpy().astype(float) for k in
             ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
base = (base[0].ravel(), *base[1:])
rng = np.random.default_rng(51001)
models = [base]+[tuple(w+s*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape)
                      for w in base) for s in (.0025, .02)]
geometry, sha = load_geometry('/workspace/clean-neurons/m16/approaches/small_h2_56/geometry.py')
data = dict(geometry_sha256=sha, models=[], comparison='same process and rays; alternating order')
data['source_sha256'] = {name: hashlib.sha256((Path(signed_l1_triangles.__code__.co_filename).parent/name).read_bytes()).hexdigest()
                         for name in ('triangle_batch.py', 'triangle.py', 'positive.py', 'moments.py')}
for i, model in enumerate(models):
    before = time.process_time()
    g = geometry.Geometry(seconds=110-(before-start))
    rays = g.simplices(g.correct(model, g.gates(model)))
    geometry_cpu = time.process_time()-before
    row = dict(sigma=(0., .0025, .02)[i], geometry_cpu=geometry_cpu)
    order = ('triangle', 'previous') if i % 2 == 0 else ('previous', 'triangle')
    row['execution_order'] = order
    for method in order:
        before = time.process_time()
        if method == 'triangle':
            result = integrate(rays, signed_l1_triangles, start+110)
        else:
            previous = [adaptive_mass(r) for r in rays]
            result = dict(estimate=sum(x[0] for x in previous),
                          integration_only_bound=sum(x[1] for x in previous),
                          nodes=sum(x[2]['order']**3 for x in previous))
        result['measured_integration_cpu'] = time.process_time()-before
        result['production_cpu_including_geometry'] = geometry_cpu+result['measured_integration_cpu']
        row[method] = result
    row['estimate_difference'] = row['triangle']['estimate']-row['previous']['estimate']
    row['total_speed_ratio'] = row['previous']['production_cpu_including_geometry']/row['triangle']['production_cpu_including_geometry']
    row['integration_speed_ratio'] = row['previous']['measured_integration_cpu']/row['triangle']['measured_integration_cpu']
    data['models'].append(row)
    data['total_cpu'] = time.process_time()-start
    (HERE/'cost_results.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(dict(sigma=row['sigma'], geometry_cpu=geometry_cpu,
                         total_ratio=row['total_speed_ratio'], integration_ratio=row['integration_speed_ratio'],
                         difference=row['estimate_difference'])), flush=True)
