"""T205: exact region counts. Runs estT exact mode with mass() stubbed to 0 (the exact splitting never uses masses), so
only geometry is paid. Two counts of distinct all-step gate patterns over the exact leaf cells:
  certified: forward every ray of the leaf; per (step, neuron) the gate is on iff max over rays of the pre-activation
             > TOL * |row| scale; a leaf where some (step, neuron) has rays strictly on both sides counts as a violation
             (gates not constant on the leaf; should be 0 for exact leaves).
  center:    forward the mean of the leaf's rays (the T194 run.py hook).
usage: python regions_exact.py H T name -> regions/H{H}_T{T}_<name>.json (resumable: skips if present)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo
from leaderboard import _noisy
H, T, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
fn = os.path.join(HERE, "regions", f"H{H}_T{T}_{name}.json")
if os.path.exists(fn):
    print("exists", fn); sys.exit(0)
m = zoo(H, T, int(name[3])) if "noisy" not in name else _noisy(zoo(H, T, 0), 4, 0.005, 7)[int(name[-1])]
Wi, Wh, Wo = m
CERT, CENTER = set(), set(); VIOL = [0]; TOL = 1e-9


def patterns(P):
    h = np.zeros((len(P), H)); G = []; bad = False
    for t in range(T):
        pre = h @ Wh.T + P[:, t:t + 1] * Wi[None]
        sc = TOL * (np.abs(pre).max() + 1e-300)
        bad |= bool(np.any((pre.max(0) > sc) & (pre.min(0) < -sc)))
        G.append(pre.max(0) > sc); h = np.maximum(pre, 0)
    return np.packbits(np.concatenate(G)).tobytes(), bad


def ce(c, g, A, Wo_, truth, ops):
    estT.STATS["cells"] += 1
    p, bad = patterns(c.R); CERT.add(p); VIOL[0] += bad
    q = c.R.mean(0)[None]; h = np.zeros((1, H)); G = []
    for t in range(T):
        h = np.maximum(h @ Wh.T + q[:, t:t + 1] * Wi[None], 0); G.append(h[0] > 0)
    CENTER.add(np.packbits(np.concatenate(G)).tobytes())
    return 0.0


estT._cell_error = ce; estT.mass = lambda c, ops: 0.0
estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0)
o = Ops(); t0 = time.process_time(); estT.estT(m, o, mode="exact")
r = dict(regions_certified=len(CERT), regions_center=len(CENTER), cert_violations=VIOL[0], geometry_ops=o.n,
         cpu=time.process_time() - t0, **estT.STATS)
os.makedirs(os.path.dirname(fn), exist_ok=True); json.dump(r, open(fn, "w"), indent=1)
print(H, T, name, r, flush=True)
