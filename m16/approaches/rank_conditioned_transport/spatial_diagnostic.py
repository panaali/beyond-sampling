"""Independent-density local interpolation checks, not global error bounds."""
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri
from engine import batched_terminal, possible_labels

HERE = Path(__file__).resolve().parent
source = json.loads((HERE/'spatial_results.json').read_text())
model = tuple(np.array(a) for a in source['weights'])
start = time.process_time()
stats = defaultdict(float)
nodes, weights = leggauss(8)
q = np.r_[(nodes+1)/4, .5+(nodes+1)/4]
weights = np.tile(weights/4, 2)*12*q*q*(1-q)
alpha = ndtr(-6.)
rows = []
for row in source['rows']:
    bounds = np.array(row['stats']['stage_bounds'][-2])*(1+1e-12)
    if row['mode'] == 'baseline':
        axes = [b*np.linspace(0, 1, row['h'])**2 for b in bounds]
    else:
        axes = [np.array(m['axis']) for m in row['stats']['mesh_metadata']['monitor']]
    results = []
    for d in range(2):
        edges = axes[d]
        positions = np.stack((edges[:-1], (edges[:-1]+edges[1:])/2, edges[1:]), axis=1)
        grid = np.empty((len(edges)-1, 3, 17, 2))
        grid[..., d] = positions[..., None]
        grid[..., 1-d] = bounds[1-d]*np.linspace(0, 1, 17)**2
        grid = grid.reshape(-1, 2)
        cache = {}
        residuals = np.zeros(len(edges)-1)
        max_residual = 0.
        labels = [j for j in possible_labels(model[2]) if j != 3]
        for label in labels:
            for qq, weight in zip(q, weights):
                u = ndtri(alpha+(1-2*alpha)*qq)
                vals = batched_terminal(model, grid, u, qq, label, 6., stats, cache).reshape(len(edges)-1, 3, 17, 2)
                error = abs(vals[:, 1]-(vals[:, 0]+vals[:, 2])/2)
                residuals += weight*error.mean(axis=(1, 2))/len(labels)
                max_residual = max(max_residual, float(error.max()))
        prior_mass = np.diff(np.sqrt(edges/bounds[d]))
        results.append(dict(weighted_residual=float(prior_mass@residuals),
                            max_residual=max_residual, bin_residuals=residuals.tolist()))
    rows.append(dict(h=row['h'], mode=row['mode'], dimensions=results))
    print(json.dumps(dict(h=row['h'], mode=row['mode'], metric=[r['weighted_residual'] for r in results])), flush=True)
(HERE/'spatial_diagnostic_results.json').write_text(json.dumps(dict(rows=rows, work=dict(stats), cpu_seconds=time.process_time()-start), indent=2)+'\n')
