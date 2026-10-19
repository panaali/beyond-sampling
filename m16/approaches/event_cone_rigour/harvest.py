"""Harvest step-10 event regions (tri, full gate pattern) from line-walk event points (T373 event_pass.lines,
re-implemented here). usage: python harvest.py p1,p2,p3 N_LINES -> data/regions_<tri>.npz"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import common  # noqa: E402

T0 = 1.5; G = 20; U = (np.arange(G) + .5) / G


def lines(rng, m, perm):
    """T373's line walk: sorted N(0,1) draws, g23 pushed past T0, x_(3) walked on a 20-point grid."""
    z = np.sort(rng.standard_normal((m, 10)), 1)[:, ::-1]
    z[:, 1] = np.maximum(z[:, 1], z[:, 3] + T0 + rng.exponential(.5, m)); z[:, 0] = np.maximum(z[:, 0], z[:, 1])
    zz = np.repeat(z, G, 0); zz[:, 2] = np.repeat(z[:, 3], G) + np.tile(U, m) * np.repeat(z[:, 1] - T0 - z[:, 3], G)
    P = np.repeat(perm, G, 0); X = np.zeros_like(zz); np.put_along_axis(X, P, zz, 1)
    hs = common.run(X); pick = (hs[:, -1] @ common.Wo.T).argmax(1)
    return X, hs, pick == P[:, 2]


def main():
    tri = tuple(int(v) for v in sys.argv[1].split(",")); n = int(sys.argv[2])
    rng = np.random.default_rng(400 + 100 * tri[0] + 10 * tri[1] + tri[2])
    lows = np.array([i for i in range(10) if i not in tri]); pts = {}
    for _ in range(max(1, n // 10_000)):
        m = 10_000; perm = np.empty((m, 10), int); perm[:, :3] = tri
        perm[:, 3:] = lows[np.argsort(rng.random((m, 7)), 1)]
        X, hs, hit = lines(rng, m, perm)
        for x, h in zip(X[hit], hs[hit, 1:] > 0):
            pts.setdefault(np.packbits(h).tobytes(), x)
    keys = list(pts); gates = np.array([np.unpackbits(np.frombuffer(k, np.uint8))[:160].reshape(10, 16) for k in keys])
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    np.savez(os.path.join(HERE, "data", f"regions_{'-'.join(map(str, tri))}.npz"), gates=gates,
             x=np.array([pts[k] for k in keys]), tri=np.array(tri))
    print(tri, "event regions", len(keys))


if __name__ == "__main__":
    main()
