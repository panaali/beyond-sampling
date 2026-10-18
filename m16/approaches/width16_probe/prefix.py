"""T216/T217: exact gate-pattern regions of the prefixes of M_16,10 (k = first k steps, a bias-free ReLU map of R^k).
Port of T192's gpu_cost_law/prefix.py (same start: the k+1 cones of a regular-simplex fan of R^k, each cut by x_0 = 0;
then estT exact `_margin_split` at steps 2..k), single process, pure numpy (this folder's estT copy), plus an optional
per-leaf record (leaf cone + gate history) for T217. Models: target | sym (symmetrised by symm_model in this file).
usage: python prefix.py model k [save]   -> prefix/<model>_k{k}.json (+ .npz leaves when save)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops


def target():
    from alg_zoo import example_2nd_argmax
    m = example_2nd_argmax().eval()
    f = lambda t: t.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)


GROUP, ANTI = (1, 6, 7, 8), (4,)


def symm_model(model, group=GROUP, anti=ANTI):
    """Project the incoming vectors u_j = [W_ih[j], W_hh[j,:]] of group + anti onto their common principal direction
    (sign-aligned; anti enters negated), u_j <- (u_j . d) d. Then the group's pre-activations are exactly proportional,
    so their gates coincide and the anti neurons' gates are exactly complementary. Outgoing weights unchanged."""
    Wi, Wh, Wo = (W.copy() for W in model); idx = list(group) + list(anti)
    U = np.c_[Wi, Wh][idx]; s = np.array([1.0] * len(group) + [-1.0] * len(anti))
    Un = s[:, None] * U / np.linalg.norm(U, axis=1, keepdims=True)
    d = np.linalg.svd(Un)[2][0]; d *= np.sign(d @ Un.sum(0))
    for q, j in enumerate(idx):
        u = (U[q] @ d) * d; Wi[j], Wh[j] = u[0], u[1:]
    return Wi, Wh, Wo


def start_cells(k, Wi):
    V = np.vstack([np.eye(k), -np.ones((1, k)) / np.sqrt(k)]); V = V - V.mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
    e0 = np.eye(k)[0]; out = []
    for i in range(k + 1):
        c = estT.Cone.from_normals(np.linalg.inv(np.delete(V, i, 0).T))
        for sigma, part in zip((1.0, -1.0), estT.clip(c, e0, Ops())):
            if part is not None:
                out.append((part, (sigma * Wi > 0).astype(float), np.outer(Wi, e0)))
    return out


def run(model, k, save=False):
    Wi, Wh, _ = model; E = np.eye(k); ops = Ops(); pats = set(); leaves = []; cells = 0; t0 = time.process_time()
    stack = [(c, g, A, 1, np.packbits(g > 0).tobytes()) for c, g, A in start_cells(k, Wi)]
    while stack:
        c, g, A, t, hist = stack.pop()
        if t == k:
            cells += 1; pats.add(hist)
            if save:
                leaves.append((c, hist))
            continue
        At = Wh @ (g[:, None] * A) + np.outer(Wi, E[t]); ops.add(len(Wi) * int(g.sum()) * t)
        for c2, g2 in estT._margin_split(c, At, model, t, 0, E[0], 0.1, ops, True):
            stack.append((c2, g2, At, t + 1, hist + np.packbits(g2 > 0).tobytes()))
    return dict(k=k, cells=cells, regions=len(pats), ops_geometry=ops.n, cpu=time.process_time() - t0), leaves


if __name__ == "__main__":
    name, k = sys.argv[1], int(sys.argv[2]); save = len(sys.argv) > 3
    model = target() if name == "target" else symm_model(target())
    r, leaves = run(model, k, save)
    os.makedirs(os.path.join(HERE, "prefix"), exist_ok=True); fn = os.path.join(HERE, "prefix", f"{name}_k{k}")
    json.dump(r, open(fn + ".json", "w"), indent=1)
    if save:
        np.savez_compressed(fn + ".npz", rays=np.array([l[0].R for l in leaves], dtype=object), hist=np.array([l[1] for l in leaves], dtype=object))
    print(name, r, flush=True)
