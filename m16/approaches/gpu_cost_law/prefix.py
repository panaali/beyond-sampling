"""T192 / DRI top priority: exact gate-pattern regions of the PREFIXES of the target M_16,10 (example_2nd_argmax).

Prefix k = the first k recurrent steps, a bias-free ReLU map of x_0..x_{k-1} in R^k. Geometry only (no masses, no
readout): start from a pointed fan of R^k (the k+1 cones of a regular simplex) split by x_0 = 0 (step-1 gate planes),
then at steps 2..k split every cell by every cutting gate (estT exact mode, `_margin_split`, unchanged geometry +
numba hot spots). Leaves = cells; distinct gate histories = activation regions (exact count). The fan and x_0 cuts
only add cells, never regions.

usage: python prefix.py k [workers] [maxsec]   -> results/prefix/k{k}.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "understand_m43"), os.path.join(HERE, "..", "arc_m43_repro"),
                os.path.join(HERE, "..", "..")]
import estG  # noqa: E402,F401
import estT_base as B  # noqa: E402
if os.environ.get("T192_FAST", "1") == "1":
    import geomfast  # noqa: E402,F401
from m43 import Ops  # noqa: E402

M = {}


def target():
    from alg_zoo import example_2nd_argmax
    m = example_2nd_argmax().eval()
    Wi = m.rnn.weight_ih_l0.detach().numpy()[:, 0].astype(np.float64)
    Wh = m.rnn.weight_hh_l0.detach().numpy().astype(np.float64)
    Wo = m.linear.weight.detach().numpy().astype(np.float64)
    return Wi, Wh, Wo


def start_cells(k):
    """(cone, g1, A1) for the simplex fan of R^k cut by x_0 = 0."""
    Wi = M["model"][0]
    V = np.vstack([np.eye(k), -np.ones((1, k)) / np.sqrt(k)])          # k+1 vectors positively spanning R^k
    V = V - V.mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
    e0 = np.eye(k)[0]; out = []
    for i in range(k + 1):
        R = np.delete(V, i, 0)
        c = B.Cone.from_normals(np.linalg.inv(R.T))
        for sigma, part in zip((1.0, -1.0), B.clip(c, e0, Ops())):
            if part is not None:
                out.append((part, (sigma * Wi > 0).astype(float), np.outer(Wi, e0)))
    return out


def _children(c, g, A, t, ops):
    Wi, Wh, _ = M["model"]; k = M["k"]
    At = Wh @ (g[:, None] * A) + np.outer(Wi, np.eye(k)[t]); ops.add(len(Wi) * int(g.sum()) * t)
    return [(c2, g2, At) for c2, g2 in B._margin_split(c, At, M["model"], t, 0, np.eye(k)[0], 0.1, ops, True)]


class _Col:
    """collects the simplices of leaf cells with their leaf index (masses mode)"""
    def __init__(self):
        self.R = []; self.I = []; self.cur = 0

    def add(self, R):
        self.R.append(R.copy()); self.I.append(self.cur)


def _dfs(item):
    c, g, A, t, hist = item
    ops = Ops(); pats = set(); cells = 0; stack = [(c, g, A, t, hist)]; t0 = time.time()
    hists = []; col = _Col(); estG.BUF[0] = col
    for key in B.STATS:
        B.STATS[key] = 0
    while stack:
        c, g, A, t, hist = stack.pop()
        if t == M["k"]:
            cells += 1; pats.add(hist)
            if M.get("masses"):
                col.cur = len(hists); hists.append(hist); B.mass(c, Ops())
            continue
        for c2, g2, A2 in _children(c, g, A, t, ops):
            stack.append((c2, g2, A2, t + 1, hist + np.packbits(g2 > 0).tobytes()))
    if M.get("masses"):
        pats = (hists, np.array(col.R).reshape(-1, M["k"], M["k"]) if col.R else np.zeros((0, M["k"], M["k"])),
                np.array(col.I, dtype=np.int64))
    return cells, pats, ops.n, dict(B.STATS), time.time() - t0


def _init(model, k, masses=False):
    M.update(model=model, k=k, masses=masses)


def run(k, workers=11, maxsec=None, masses=False, device="cpu"):
    model = target(); _init(model, k, masses)
    RM = {}; leafsum = 0.0
    t0 = time.time(); ops = Ops()
    front = [(c, g, A, 1, np.packbits(g > 0).tobytes()) for c, g, A in start_cells(k)]
    while front and front[0][3] < k and len(front) < 64 * workers:     # breadth-first until enough subtrees
        nxt = []
        for c, g, A, t, hist in front:
            for c2, g2, A2 in _children(c, g, A, t, ops):
                nxt.append((c2, g2, A2, t + 1, hist + np.packbits(g2 > 0).tobytes()))
        front = nxt
    perm = np.random.default_rng(0).permutation(len(front)); front = [front[i] for i in perm]   # random subtree order
    cells = 0; pats = set(); gops = ops.n; sec_geo = time.time() - t0; st = dict(cut=0, resolved=0); done = 0
    with mp.get_context("fork").Pool(workers, _init, (model, k, masses)) as pool:
        for n, p, o, s, sec in pool.imap_unordered(_dfs, front, chunksize=1):
            if masses:
                import gmass
                hists, Rs, Is = p
                cm = gmass.grouped(Rs, Is, len(hists), device) if len(Rs) else np.zeros(len(hists))
                for h, mv in zip(hists, cm):
                    RM[h] = RM.get(h, 0.0) + float(mv)
                leafsum += float(cm.sum()); p = set(hists)
            cells += n; pats |= p; gops += o; sec_geo += sec; done += 1
            for key in st:
                st[key] += s.get(key, 0)
            if maxsec and time.time() - t0 > maxsec:
                pool.terminate(); break
    r = dict(k=k, complete=done == len(front), subtrees=len(front), subtrees_done=done, cells=cells,
             regions=len(pats), ops_geometry=gops, sec_cpu=sec_geo, sec_wall=time.time() - t0, workers=workers, **st)
    if masses:
        r["leafsum_minus_1"] = leafsum - 1.0
        m = np.sort(np.array(list(RM.values())))[::-1]; cum = np.cumsum(m)
        r["N_delta"] = {f"{dl:g}": int(np.searchsorted(cum, cum[-1] * (1 - dl)) + 1) for dl in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)}
        r["mass_eff_regions"] = float(1.0 / np.sum((m / m.sum()) ** 2))       # inverse Simpson (effective count)
        keys = list(RM)
        np.savez_compressed(os.path.join(HERE, "results", "prefix", f"k{k}_regionmass.npz"),
                            patterns=np.array([np.frombuffer(h, np.uint8) for h in keys]),
                            mass=np.array([RM[h] for h in keys]), complete=r["complete"])
    if not r["complete"]:
        r["note"] = ("stopped at maxsec: cells/regions are lower bounds; subtrees were done in a seeded random order, "
                     "so cells_est = cells * subtrees / subtrees_done is an unbiased estimate of total cells")
        r["cells_est"] = cells * len(front) / max(done, 1)
    return r


if __name__ == "__main__":
    k = int(sys.argv[1]); w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ms = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "0" else None
    masses = len(sys.argv) > 4 and sys.argv[4] == "masses"; dev = sys.argv[5] if len(sys.argv) > 5 else "cpu"
    os.makedirs(os.path.join(HERE, "results", "prefix"), exist_ok=True)
    r = run(k, w, ms, masses, dev)
    json.dump(r, open(os.path.join(HERE, "results", "prefix", f"k{k}{'_masses' if masses else ''}.json"), "w"), indent=1)
    print(json.dumps(r))
