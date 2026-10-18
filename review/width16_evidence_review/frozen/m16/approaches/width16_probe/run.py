"""T194: frozen recipe108 (eps .1, as estT_fix/rescore.py) and exact enumeration on zoo (H,T), with leaf-mass sum.
usage: python run.py H T name mode   (name zoo<s> | zoo0_noisy<i>; mode recipe108 | exact) -> res/H{H}_T{T}_<name>_<mode>.json
Noisy copies as estT_fix: leaderboard._noisy(zoo(H,T,0), 4, 0.005, 7). Resumable: skips if the output exists.
Abort guard: raises Stop when ops > 1e11 or CPU time > 60 min (recorded in the json as stopped)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo
from leaderboard import _noisy
H, T, name, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
fn = os.path.join(HERE, "res", f"H{H}_T{T}_{name}_{mode}.json")
if os.path.exists(fn):
    print("exists", fn); sys.exit(0)
m = zoo(H, T, int(name[3])) if "noisy" not in name else _noisy(zoo(H, T, 0), 4, 0.005, 7)[int(name[-1])]
MAX_OPS, MAX_CPU = 1e11, 3600.0


class Stop(Exception):
    pass


class GOps(Ops):
    def add(self, k):
        super().add(k)
        if self.n > MAX_OPS:
            raise Stop("ops")


LEAF = [0.0]; ce0 = estT._cell_error; c0 = time.process_time()


PAT = set()


def pattern(p):
    """gate pattern of all steps at point p (diagnostic, no ops): exact-mode leaves have every gate resolved, so the
    number of distinct leaf patterns is the exact number of activation regions (the regions_lb denominator)."""
    Wi, Wh, _ = m; h = np.zeros(len(Wi)); G = []
    for t in range(T):
        h = np.maximum(Wh @ h + p[t] * Wi, 0); G.append(h > 0)
    return np.packbits(np.concatenate(G)).tobytes()


def ce(c, g, A, Wo, truth, ops):
    if time.process_time() - c0 > MAX_CPU:
        raise Stop("cpu")
    if mode == "exact":
        PAT.add(pattern(c.R.mean(0)))
    LEAF[0] += estT.mass(c, Ops()); return ce0(c, g, A, Wo, truth, ops)


estT._cell_error = ce
estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0)
o = GOps(); t0 = time.time()
try:
    acc = estT.estT(m, o, mode=mode); stop = None
except Stop as e:
    acc = None; stop = str(e)
r = dict(acc=acc, ops=o.n, sec=time.time() - t0, cpu=time.process_time() - c0, stopped=stop,
         leafsum_minus_1=None if stop else LEAF[0] - 1,
         regions_exact=len(PAT) if mode == "exact" and not stop else None, **estT.STATS)
os.makedirs(os.path.dirname(fn), exist_ok=True); json.dump(r, open(fn, "w"), indent=1)
print(H, T, name, mode, {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
