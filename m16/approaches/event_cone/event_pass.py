"""T373 event pass (post-hoc, not pre-registered as such; the pre-registered proposal found no event points).
(a) Global: where do far-gap rank-3 picks (t0 = 1.5) live? Sorted N(0,1)^10 draws with g23 pushed past t0, a random
    assignment of ranks to positions, and a 20-point grid of x_p3 on (v4, v2 - t0) (T340's e_p3 line). Hit counts per
    triple (line points, NOT probability-weighted: a diagnostic of support, not of mass).
(b) Per triple: the same line walk with the ranks fixed to the triple; counts event points and their distinct
    k-step gate prefixes (k = 5, 6). Distinct event prefixes / count_k is a sampled LOWER bound on the
    event-feasible region fraction at k.
usage: python event_pass.py -> results_event_pass.json"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)
import common  # noqa: E402
from cone_count import TRIPLES  # noqa: E402

T0 = 1.5; G = 20; u = (np.arange(G) + .5) / G


def lines(rng, m, perm):
    z = np.sort(rng.standard_normal((m, 10)), 1)[:, ::-1]
    z[:, 1] = np.maximum(z[:, 1], z[:, 3] + T0 + rng.exponential(.5, m)); z[:, 0] = np.maximum(z[:, 0], z[:, 1])
    zz = np.repeat(z, G, 0); zz[:, 2] = np.repeat(z[:, 3], G) + np.tile(u, m) * np.repeat(z[:, 1] - T0 - z[:, 3], G)
    P = np.repeat(perm, G, 0); X = np.zeros_like(zz); np.put_along_axis(X, P, zz, 1)
    hs = common.run(X); pick = (hs[:, -1] @ common.Wo.T).argmax(1)
    return X, P, hs, pick == P[:, 2]


def main():
    out = {}
    if "--no-global" in sys.argv:                                  # global part already in logs/event_pass.log
        return per_triple(out)
    rng = np.random.default_rng(373); C = collections.Counter(); n = 0
    for _ in range(200):                                           # 4e7 line points
        m = 10_000; perm = np.argsort(rng.random((m, 10)), 1)
        X, P, hs, hit = lines(rng, m, perm); n += len(X)
        for r in P[hit][:, :3]:
            C[str(tuple(int(v) for v in r))] += 1
    tot = sum(C.values())
    out["global"] = dict(line_points=n, hits=tot, triples_with_hits=len(C), top=C.most_common(30),
                         top6_share=sum(c for _, c in C.most_common(6)) / max(tot, 1))
    print(out["global"], flush=True)
    per_triple(out)


def per_triple(out):
    for tri in TRIPLES:
        rng = np.random.default_rng(1000 + 100 * tri[0] + 10 * tri[1] + tri[2])
        lows = np.array([i for i in range(10) if i not in tri])
        ev = 0; n = 0; pre = {5: set(), 6: set()}
        for _ in range(20):                                        # 4e6 line points per triple
            m = 10_000; perm = np.empty((m, 10), int); perm[:, :3] = tri
            perm[:, 3:] = lows[np.argsort(rng.random((m, 7)), 1)]       # lows' ranks assigned at random
            X, P, hs, hit = lines(rng, m, perm); n += len(X); ev += int(hit.sum())
            for k in (5, 6) if hit.any() else ():
                for r in np.packbits(hs[hit, 1:k + 1] > 0, axis=2).reshape(hit.sum(), -1):
                    pre[k].add(r.tobytes())
        out[str(tri)] = dict(line_points=n, event_points=ev, event_prefixes_k5=len(pre[5]), event_prefixes_k6=len(pre[6]))
        print(tri, out[str(tri)], flush=True)
    json.dump(out, open(os.path.join(HERE, "results_event_pass_triples.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
