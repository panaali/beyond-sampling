"""Print the closed-form mechanism quantities of an M_4,3 (README section "What the network does and why it errs").
All numbers are formulas of (Wi, Wh, Wo); nothing is fitted. usage: python mechanism.py [zoo_seed]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
from m43 import Wi0, Wh0, Wo0
from est_mech import _envelope, Ops

np.set_printoptions(precision=3, suppress=True)
E = np.eye(3)
model = (Wi0, Wh0, Wo0)
if len(sys.argv) > 1:
    from zoo43 import zoo_model
    model = zoo_model(int(sys.argv[1]))
Wi, Wh, Wo = model
print("Wi", Wi)
for sigma in (1.0, -1.0):
    g1 = ((sigma * Wi) > 0).astype(float)
    A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1])
    t = -A2[:, 0] / Wi
    print(f"\nchart sign(x0) = {sigma:+.0f}: step-1 active neurons {np.flatnonzero(g1)}")
    print("  step-2 switch planes x1 = t_k x0, t_k =", t, " (gate k on for x1/x0", ["above" if sigma * w > 0 else "below" for w in Wi], ")")
    ts = np.sort(t); b = [-np.inf] + list(ts) + [np.inf]
    for s in range(5):
        r = (b[s] + b[s + 1]) / 2 if np.isfinite(b[s]) and np.isfinite(b[s + 1]) else (b[s + 1] - 1 if np.isfinite(b[s + 1]) else b[s] + 1)
        g2 = (sigma * (A2[:, 0] + Wi * r) > 0).astype(float)
        A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2])
        # tau_k(r) = -(A3_k0 + A3_k1 r)/A3_k2 : x2-threshold of neuron k, linear in r = x1/x0 inside the strip
        print(f"  strip {s} r in ({b[s]:.3f},{b[s+1]:.3f}) step-2 on {np.flatnonzero(g2)}: tau_k = alpha_k + beta_k r with "
              f"alpha {-A3[:, 0] / A3[:, 2]}, beta {-A3[:, 1] / A3[:, 2]}")
print("\nreadout of each pair (a,b): class order and ratio thresholds nu = h_b/h_a (upper envelope of Wo[:,a] + Wo[:,b] nu)")
for a in range(4):
    for b in range(a + 1, 4):
        env = _envelope(Wo[:, a], Wo[:, b], Ops())
        print(f"  pair ({a},{b}):", " | ".join(f"class {c} for nu in [{lo:.3f},{hi:.3f}]" for lo, hi, c in env))
print("\nasymptotic ratios when x2 dominates: u pair h1/h0 -> Wi1/Wi0 =", Wi[1] / Wi[0], "; v pair h2/h3 -> Wi2/Wi3 =", Wi[2] / Wi[3])
print("pole logits (forward pass at +-e_k); at pole e_k the two classes compared there must tie:")
from m43 import forward
for k in range(3):
    for s in (1, -1):
        l = forward(model, (s * E[k])[None])[0]
        pair = [c for c in range(3) if c != k]
        print(f"  {'+' if s > 0 else '-'}e{k}: logits {l}  gap l{pair[0]}-l{pair[1]} = {l[pair[0]] - l[pair[1]]:+.3f}")
