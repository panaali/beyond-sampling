"""Arithmetic on saved reference moments only; no model evaluation."""
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE.parent / 'whitebox_cumulants/results/references'
d = 1024
er = math.sqrt(2) * math.exp(math.lgamma((d+1)/2)-math.lgamma(d/2))
radius_variance = d-er*er
rows = []
for name in ('mini_0', 'mini_1', 'full_0', 'full_1'):
    meta = json.loads((REF / (name+'.json')).read_text())
    with np.load(REF / (name+'.npz')) as archive:
        means = archive['final_means'].astype(float)
    variance = meta['avg_variance']
    second_moment = variance + float(np.mean(means**2))
    removed = radius_variance/d * second_moment
    rows.append(dict(row=name, avg_variance=variance,
                     avg_second_moment=second_moment,
                     estimated_removed_variance=removed,
                     remaining_variance_ratio=1-removed/variance,
                     ideal_equal_sample_gain=variance/(variance-removed)))
report = dict(width=d, radius_mean=er, radius_variance=radius_variance,
              rows=rows, model_forward_passes=0,
              scope='Saved public moment arithmetic; normalization cost omitted, '
                    'finite reference uncertainty retained; not an estimator run.')
(HERE / 'radial_screen.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
