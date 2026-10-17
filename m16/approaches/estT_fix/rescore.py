"""T164: frozen recipe108 (eps .1, as final3/run.py) with the T131-fixed estT, scored against fixed exact mode, plus the
per-step over-split diagnosis (DIAG hook; 64 Dirichlet points + rays of the minority side per resolved split, no ops).
usage: python rescore.py name [tight]   -> rescore/<name>[_tight].json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo
from leaderboard import fpe_flops, _noisy
name = sys.argv[1]; TIGHT = len(sys.argv) > 2
m = zoo(8, 4, int(name[3])) if "noisy" not in name else _noisy(zoo(8, 4, 0), 4, 0.005, 7)[int(name[-1])]
if TIGHT:
    import tight; tight.install(estT)
rng = np.random.default_rng(1); EV = []


def diag(c, A, g, k, t, model, truth):
    Wi, Wh, Wo = model; T = Wo.shape[0]
    ins, out = estT.clip(c, A[k], Ops()); mino = out if g[k] > 0 else ins
    if mino is None:
        EV.append((t, 0, 0)); return
    P = np.vstack([mino.R, rng.dirichlet(np.ones(len(mino.R)), 64) @ mino.R])
    pre = P @ A.T; res = []
    for frz in (False, True):
        G = (pre > 0).astype(float)
        if frz:
            G[:, k] = g[k]
        h = pre * G
        for s in range(t + 1, T):
            h = np.maximum(h @ Wh.T + P[:, s:s + 1] * Wi[None], 0)
        res.append((h @ Wo.T).argmax(1))
    EV.append((t, int(np.any(res[0] != res[1])), int(np.any((res[0] == truth) != (res[1] == truth)))))


r = {}
for mode in (["exact"] if not TIGHT else []) + ["recipe108"]:
    estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0); EV.clear()
    estT.DIAG[0] = diag if mode == "recipe108" else None
    o = Ops(); t0 = time.time(); acc = estT.estT(m, o, mode=mode)
    r[mode] = dict(acc=acc, ops=o.n, cells=estT.STATS["cells"], resolved=estT.STATS["resolved"], sec=time.time() - t0)
    if mode == "recipe108":
        E = np.array(EV).reshape(-1, 3)
        r["splits"] = {int(t): dict(n=int((E[:, 0] == t).sum()), class_changes=int(E[E[:, 0] == t, 1].sum()),
                                    err_changes=int(E[E[:, 0] == t, 2].sum())) for t in np.unique(E[:, 0])}
os.makedirs(os.path.join(HERE, "rescore"), exist_ok=True)
json.dump(r, open(os.path.join(HERE, "rescore", name + ("_tight" if TIGHT else "") + ".json"), "w"), indent=1)
print(name, {k: (round(v["acc"], 9), f"{v['ops']:.3g}", v["cells"]) for k, v in r.items() if "acc" in v}, r["splits"])
