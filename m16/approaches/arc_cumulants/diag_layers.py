"""Diagnosis only (uses random inputs for the reference): layer-by-layer error of ARC's propagated means and
variances on M_H,T, unconditional mode. Usage: python diag_layers.py H T K [kind]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "m16")
import numpy as np
import arc_kprop as A
import leaderboard as LB

H, T, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
kind = sys.argv[4] if len(sys.argv) > 4 else "SIMPLE"
name, m = LB._base(H, T)
layers = A.rnn_layers(m) + A.readout_uncond(m)
tr = A.propagate_trace(layers, A.uncond_input(H, T, K), K, kind)
x = np.random.default_rng(0).standard_normal((400_000, T))
s = np.concatenate([np.zeros((len(x), H)), x], 1)
labels = [f"h_{t+1}" for t in range(T)] + ["R1 pairwise", "R2 counts", "R3 cells", "output"]
for li, ((W, b, act), (mu, var)) in enumerate(zip(layers, tr)):
    z = s @ W.T + (0 if b is None else b)
    if act is None:
        s = z
    else:
        a = np.array(act)
        s = np.where(a == A.RELU, np.maximum(z, 0), np.where(a == A.HEAV, (z > 0).astype(float), z))
    sel = slice(0, H) if li < T else slice(None)
    mc_mu, mc_var = s.mean(0)[sel], s.var(0)[sel]
    e_mu = np.abs(mu[sel] - mc_mu)
    msg = f"{labels[li]:12s} mean err max {e_mu.max():8.3f} rms {np.sqrt((e_mu**2).mean()):8.3f} (MC scale {np.sqrt((mc_mu**2).mean()):7.3f})"
    if var is not None:
        e_v = np.abs(var[sel] - mc_var)
        msg += f" | var err rms {np.sqrt((e_v**2).mean()):8.3f} (MC scale {np.sqrt((mc_var**2).mean()):7.3f})"
    print(msg)
