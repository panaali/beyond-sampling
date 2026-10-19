"""Pre-registered tau rule: per model, grid taus with exact B closest (log10) to 1e-6 and 1e-8, kept if in [1e-9, 1e-5]."""
import numpy as np

from s3lib import DATA, Dyck2

GRID = 1 / (1 + np.exp(-np.linspace(np.log(1e-3 / (1 - 1e-3)), np.log(.999 / .001), 400)))
_W = None


def exact_B(name, taus, L=15):
    global _W
    if _W is None:
        _W = Dyck2().enumerate_small(L, as_int8=True)[1]
    mx = np.load(f"{DATA}/{name}_L15_max.npy").astype(np.float64)
    o = np.argsort(mx); ms = mx[o]; cw = np.cumsum(_W[o][::-1])[::-1]  # cw[i] = mass of sequences >= ms[i]
    out = []
    for t in taus:
        i = np.searchsorted(ms, t, side="right")
        out.append(float(cw[i]) if i < len(ms) else 0.0)
    return np.array(out)


def points(name):
    B = exact_B(name, GRID)
    pts = []
    for target in (1e-6, 1e-8):
        ok = (B >= 1e-9) & (B <= 1e-5)
        if not ok.any():
            continue
        i = np.argmin(np.where(ok, np.abs(np.log10(np.maximum(B, 1e-300)) - np.log10(target)), np.inf))
        if all(abs(GRID[i] - p[0]) > 0 for p in pts):
            pts.append((float(GRID[i]), float(B[i]), target))
    return pts
