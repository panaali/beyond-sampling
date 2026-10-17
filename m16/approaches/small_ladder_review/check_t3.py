"""Independent octant-first geometry versus the two-input conditional tail."""
from pathlib import Path
import importlib.util
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
APPROACHES = HERE.parent
sys.path.insert(0, str(APPROACHES / 'feynman_fundamentals'))
from tail import conditional_probability, prefix_summary

spec = importlib.util.spec_from_file_location('independent_facets', APPROACHES / 'facet_transport' / 'probe.py')
facets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facets)

def candidate(model):
    values = []
    for sign in (-1., 1.):
        h, top = prefix_summary(model, np.array([sign]))
        values.append(conditional_probability(model, h, top, 1)[0])
    return float(np.mean(values))

def reference(model):
    regions, _ = facets.gate_complex(model)
    return facets.accuracy(facets.rank_complex(regions), model[2])

def main():
    import torch
    checkpoint = Path('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    base = tuple(state[key].numpy().astype(float) for key in ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
    base = (base[0].ravel(), base[1], base[2])
    cases = [('trained_seed0', base)]
    rng = np.random.default_rng(52002)
    for sigma in (.0025, .02):
        for copy in range(4):
            model = tuple(w + sigma * np.sqrt(np.mean(w*w)) * rng.standard_normal(w.shape) for w in base)
            cases.append((f'relative_noise_{sigma}_copy{copy}', model))
    cases.append(('zero_readout', (*base[:2], np.zeros_like(base[2]))))
    cases.append(('identical_nonzero_rows', (*base[:2], np.tile(np.array([1., -2.]), (3, 1)))))
    tied = base[2].copy()
    tied[1] = tied[0]
    cases.append(('duplicate_first_two_rows', (*base[:2], tied)))
    start = time.process_time()
    rows = []
    for name, model in cases:
        a, b = candidate(model), reference(model)
        row = dict(case=name, conditional_tail=a, octant_geometry=b, absolute_difference=abs(a-b))
        if name in ('zero_readout', 'identical_nonzero_rows'):
            row['closed_form'] = 1/3
            assert abs(a-1/3) < 1e-12
        assert abs(a-b) < 1e-10, row
        rows.append(row)
    report = dict(cases=rows, cpu_seconds=time.process_time()-start,
                  compute='24 analytic conditional tails and 12 independent cone enumerations; zero sampled model inputs',
                  scope='Independent gate/rank traversal, shared spherical area identity; no full floating-point certificate or cost confirmation')
    (HERE / 't3_review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows), maximum_difference=max(r['absolute_difference'] for r in rows), cpu_seconds=report['cpu_seconds'])))

if __name__ == '__main__':
    main()
