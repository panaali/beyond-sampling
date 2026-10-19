"""T386 checks of smooth.py: (i) as eps -> 0 the value and gradient approach the hard K3 ones (dealias.est_grad);
(ii) the analytic gradient equals a central FD of the smoothed estimate itself (3 random directions)."""
import sys

import numpy as np

import common
import dealias
import smooth
import t372_rnn as rnn

nm = sys.argv[1]; m, t0, U, D = common.case(nm); w0 = rnn.flat(m)
hard = dealias.est_grad(m, t0); print("hard est", hard["est"], flush=True)
for eps in [float(e) for e in sys.argv[2].split(",")]:
    r = smooth.est_grad(m, t0, eps)
    print(f"eps {eps}: est {r['est']:.5e} (hard {hard['est']:.5e}); cos(g, g_hard) {r['grad'] @ hard['grad'] / np.linalg.norm(r['grad']) / np.linalg.norm(hard['grad']):.4f}; "
          f"|g|/|g_hard| {np.linalg.norm(r['grad']) / np.linalg.norm(hard['grad']):.3f}; pts {r['n_pts']}; C {r['C_walk']:.3g}+{r['C_grad']:.3g}", flush=True)
if len(sys.argv) > 3:
    eps = float(sys.argv[3]); r = smooth.est_grad(m, t0, eps); rng = np.random.default_rng(386900)
    for i in range(3):
        u = rng.normal(size=len(w0)); u /= np.linalg.norm(u); h = 1e-6
        ep = smooth.est_grad(rnn.unflat(w0 + h * u, common.SH), t0, eps)["est"]; em = smooth.est_grad(rnn.unflat(w0 - h * u, common.SH), t0, eps)["est"]
        print(f"FD check eps {eps} dir {i}: analytic {r['grad'] @ u:.5e} FD {(ep - em) / 2 / h:.5e}", flush=True)
