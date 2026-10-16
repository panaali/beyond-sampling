"""Check (random inputs, diagnosis only): face integrals with plain MC faces reproduce MC truth of
P(err & R23), P(err & R12), P(err & B) on a model.  usage: python check_partition.py <model> [nface]"""
import sys
import numpy as np
from math import comb
import core as C
sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/sci_feynman")
import wtool as W

nm = sys.argv[1] if len(sys.argv) > 1 else "orig"; n = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
M = W.get_model(nm); rng = np.random.default_rng(3)
# MC truth decomposition
x = rng.standard_normal((1_000_000, 10)); srt = np.sort(x, 1)[:, ::-1]
err = C.forward(M, x).argmax(1) != np.argsort(x, 1)[:, -2]
B = srt[:, 1] <= 0; R23 = ~B & (srt[:, 1] - srt[:, 2] < srt[:, 0] - srt[:, 1]); R12 = ~B & ~R23
print(f"MC: err {err.mean():.5f}  B {np.mean(err&B):.5f}  R23 {np.mean(err&R23):.5f}  R12 {np.mean(err&R12):.5f}")
# R23 faces
U = rng.random((n, 9)); u = C.q23(U[:, 0]); o1 = C.tn_above(U[:, 1], u); bath = C.tn_below(U[:, 2:], u[:, None])
cf = rng.integers(0, 360, n)
trip = [(a, b, p) for a in range(10) for b in range(a + 1, 10) for p in range(10) if p not in (a, b)]
A = np.array([t[0] for t in trip])[cf]; Bb = np.array([t[1] for t in trip])[cf]; P = np.array([t[2] for t in trip])[cf]
y = np.empty((n, 10)); r = np.arange(n); y[r, A] = u; y[r, Bb] = u; y[r, P] = o1
mask = np.ones((n, 10), bool); mask[r, A] = False; mask[r, Bb] = False; mask[r, P] = False
y[mask] = bath.ravel()
o2 = bath.max(1)
tau = np.maximum(SQ := np.sqrt(2) * np.minimum((o1 - u) / 3, u - o2), 0)
tex = np.where(u < 0, -np.sqrt(2) * u, 0.0)
m, c = C.walk_mass(M, y, A, Bb, tau, tex)
G = 2 * np.sqrt(np.pi) * m
print(f"R23 faces: {360*C.Z23*G.mean():.5f} +- {360*C.Z23*G.std()/np.sqrt(n):.5f}   cost/face {c.mean():.2f}  CV {G.std()/G.mean():.2f}")
# R12 faces
U = rng.random((n, 9)); w = C.q12(U[:, 0]); bath = C.tn_below(U[:, 1:], w[:, None])
pairs = [(a, b) for a in range(10) for b in range(a + 1, 10)]; cf = rng.integers(0, 45, n)
A = np.array([t[0] for t in pairs])[cf]; Bb = np.array([t[1] for t in pairs])[cf]
y = np.empty((n, 10)); y[r, A] = w; y[r, Bb] = w
mask = np.ones((n, 10), bool); mask[r, A] = False; mask[r, Bb] = False; y[mask] = bath.ravel()
o3 = bath.max(1)
tau = np.maximum(np.sqrt(2) * np.minimum((w - o3) / 3, w), 0)
m, c = C.walk_mass(M, y, A, Bb, tau, np.zeros(n), top=True)
G = 2 * np.sqrt(np.pi) * m
print(f"R12 faces: {45*C.Z12*G.mean():.5f} +- {45*C.Z12*G.std()/np.sqrt(n):.5f}   cost/face {c.mean():.2f}  CV {G.std()/G.mean():.2f}")
