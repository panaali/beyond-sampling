"""Independent derivative and line-mass checks for the diagnostic machinery."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr, ndtri
from probe import gradients, full_mass, anchored_rule, lp, lb

s = torch.load('/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth',
               weights_only=True, map_location='cpu')
base = (s['rnn.weight_ih_l0'].numpy()[:, 0].astype(float),
        s['rnn.weight_hh_l0'].numpy().astype(float), s['linear.weight'].numpy().astype(float))
model = lb._noisy(base, 6, .0025, seed=1)[0]
rng = np.random.default_rng(118)
x = rng.normal(size=(4, 10))
m, j = gradients(model, x, 1, 3)
fd = np.zeros_like(j)
for k in range(10):
    d = np.zeros_like(x); d[:, k] = 1e-6
    fd[:, k] = ((lb.forward(model, x+d)[:, 1]-lb.forward(model, x+d)[:, 3])
                -(lb.forward(model, x-d)[:, 1]-lb.forward(model, x-d)[:, 3]))/2e-6
assert np.max(np.abs(fd-j)) < 1e-5
checks = []
for cfg, u, mx in [(np.array([3, 4, 2]), .65, 1.63), (np.array([1, 5, 0]), .56, 1.75)]:
    lows = ndtri(np.array([[.1,.2,.3,.4,.5,.6,.7]])*ndtr(u))
    f = lp.build23(cfg[None], np.array([u]), np.array([mx]), lows)
    exact, _ = full_mass(model, f)
    n = 65536
    edges = np.linspace(-f['hi'][0], f['hi'][0], n+1)
    mid = (edges[:-1]+edges[1:])/2
    xline = f['x0'] + mid[:, None]*f['d']
    wrong = lb.forward(model, xline).argmax(1) != np.argsort(xline, axis=1)[:, -2]
    mass = (2*np.sqrt(np.pi)*np.diff(ndtr(edges/np.sqrt(2)))*wrong).sum()
    assert abs(mass-exact[0]) < 5e-5
    checks.append(dict(exact=float(exact[0]), grid=float(mass)))
out = dict(jacobian_max_abs_error=float(np.max(np.abs(fd-j))), line_mass_checks=checks,
           grid_evaluations=131072, finite_difference_evaluations=80)
Path(__file__).with_name('validation.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out))
