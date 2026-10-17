"""Separate quadrature refinement from omitted anchored higher-order interactions."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from probe import anchored_rule, lb

start = time.process_time()
p = Path(__file__).parent
s = torch.load('/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth',
               weights_only=True, map_location='cpu')
base = (s['rnn.weight_ih_l0'].numpy()[:, 0].astype(float),
        s['rnn.weight_hh_l0'].numpy().astype(float), s['linear.weight'].numpy().astype(float))
model = lb._noisy(base, 6, .0025, seed=1)[0]
rows = []
for a in json.loads((p/'results.json').read_text())['rows']:
    if a['model'] != 'dev_s0.0025_0' or a['case'] not in (2, 4):
        continue
    for q in (17, 33):
        out = anchored_rule(model, np.array(a['config']), a['tie'], a['top'], q)
        out.update(case=a['case'], q=q, truth=a['exact_full_band_mass'])
        rows.append(out)
        print(json.dumps(out), flush=True)
(p/'refinement.json').write_text(json.dumps(dict(rows=rows, cpu_seconds=time.process_time()-start), indent=2)+'\n')
