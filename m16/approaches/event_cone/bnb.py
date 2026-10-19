"""T390 kill test: branch and bound with relax.py pruning inside a triple's true event cone (README T390).
usage: python bnb.py p1,p2,p3 KMAX [--t398] -> results_bnb[398]_<tri>.json"""
import json
import os
import sys
import time

import numpy as np

import cone_count as cc
import relax
import relax2
from event_pass import lines

HERE = os.path.dirname(os.path.abspath(__file__))
T398 = "--t398" in sys.argv                                        # T398 pruner (relax2.prune_stage)
if "--delta" in sys.argv:                                          # T398b: normalisation g23 >= DELTA ||x||_inf
    relax2.DELTA = float(sys.argv[sys.argv.index("--delta") + 1])
TAG = f"_d{relax2.DELTA:.3g}" if "--delta" in sys.argv else ""


def gates_of(hist, k, fix):
    g = -np.ones((10, 16), np.int8)
    if k:
        g[:k] = np.unpackbits(np.frombuffer(hist, np.uint8)).reshape(k, 16)
    for (s, j), v in fix.items():
        g[k + s, j] = v
    return g


def event_prefixes(tri, kmax, n_lines=100_000):
    """k-prefixes (k = 1..kmax) of line-walk event points (T373 event pass, regenerated)."""
    rng = np.random.default_rng(390 + 100 * tri[0] + 10 * tri[1] + tri[2])
    lows = np.array([i for i in range(10) if i not in tri]); pre = {k: set() for k in range(1, kmax + 1)}; ev = 0
    for _ in range(n_lines // 10_000):
        m = 10_000; perm = np.empty((m, 10), int); perm[:, :3] = tri
        perm[:, 3:] = lows[np.argsort(rng.random((m, 7)), 1)]
        X, P, hs, hit = lines(rng, m, perm); ev += int(hit.sum())
        if hit.any():
            for k in pre:
                for r in np.packbits(hs[hit, 1:k + 1] > 0, axis=2).reshape(hit.sum(), -1):
                    pre[k].add(r.tobytes())
    return pre, ev


def step(cells, t, Wi, Wh, H):
    """One exact splitting step (cone_count.count's inner loop, fast path)."""
    new = []
    for G, b, xi, A, hist in cells:
        Z = Wh @ A; Z[:, t] += Wi
        pieces = [(G, xi, [])]
        for j in range(H):
            nxt = []
            for Gp, xp, g in pieces:
                d, xpos, xneg = cc.split_fast(Gp, xp, Z[j])
                if d == "off":
                    nxt.append((Gp, xp, g + [0]))
                elif d == "on":
                    nxt.append((Gp, xp, g + [1]))
                else:
                    nxt.append((np.vstack([Gp, Z[j]]), xpos, g + [1])); nxt.append((np.vstack([Gp, -Z[j]]), xneg, g + [0]))
            pieces = nxt
        for Gp, xp, g in pieces:
            g = np.array(g, float)
            new.append((Gp, None, xp, g[:, None] * Z, hist + np.packbits(g > 0).tobytes()))
    return new


def main():
    tri = tuple(int(v) for v in sys.argv[1].split(",")); kmax = int(sys.argv[2]); cap = 200_000
    Wi, Wh, _ = cc.model(); H = len(Wi)
    t0 = time.time(); pre, ev = event_prefixes(tri, kmax)
    out = dict(tri=tri, event_points=ev, event_prefixes={k: len(v) for k, v in pre.items()}, steps=[])
    print(out, round(time.time() - t0, 1), flush=True)
    G0, b0 = cc.cone(tri)
    cells = [(G0, b0, cc.interior(tri), np.zeros((H, len(Wi) and 10)), b"")]
    path = os.path.join(HERE, f"results_bnb{'398' if T398 else ''}{TAG}_{'-'.join(map(str, tri))}.json")
    for t in range(kmax):
        t1 = time.time(); new = step(cells, t, Wi, Wh, H); t2 = time.time()
        keep = []; stages = []; cstage, cstart, cleaves = [], [], []
        for c in new:
            if T398:
                lv = []; sg = relax2.prune_stage(c[0], c[3], t + 1, tri, cert=lv); stages.append(sg); inf = sg >= 0
                if inf:
                    cstage.append(sg); cstart.append(len(cleaves))
                    cleaves.extend(gates_of(c[4], t + 1, f) for f in lv)
            else:
                inf, st = relax.relax_infeasible(c[0], c[3], t + 1, tri, return_stats=True)
                stages.append(st["stage"] if inf else -1)
            if not inf:
                keep.append(c)
        kept = {c[4] for c in keep}; lost = [p for p in pre[t + 1] if p not in kept]
        st = np.array(stages)
        rec = dict(k=t + 1, children=len(new), survivors=len(keep), event_prefixes=len(pre[t + 1]),
                   event_prefixes_pruned=len(lost), pruned_by_norm=int((st == 0).sum()),
                   pruned_later=int((st > 0).sum()),
                   by_stage={int(g): int((st == g).sum()) for g in range(4)} if T398 else None, sec_split=round(t2 - t1, 1), sec_relax=round(time.time() - t2, 1))
        out["steps"].append(rec); print(rec, flush=True)
        if T398 and cstage:
            os.makedirs(os.path.join(HERE, "certs"), exist_ok=True)
            np.savez_compressed(os.path.join(HERE, "certs", f"cert{TAG}_{'-'.join(map(str, tri))}_k{t + 1}.npz"),
                                tri=np.array(tri), k=t + 1, stage=np.array(cstage, np.int8), start=np.array(cstart),
                                leaves=np.array(cleaves, np.int8), delta=relax2.DELTA)
        json.dump(out, open(path, "w"), indent=1)
        cells = keep
        if len(keep) > cap:
            print("cap", flush=True); break


if __name__ == "__main__":
    main()
