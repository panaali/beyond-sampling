"""T403: load the frozen T366 graded-g12 estimator from frozen/ (unmodified copies of claude/t218-tie-tail-labels
@5b4a1250 r1_cont/ and claude/t286-cheap-frontier @620eb47b; all 7 sha256 of t366_freeze_sha256.txt plus cf.py's
asserted at import) and split its estimate by g12 node.

est_split(model, t0s) is a memory wrapper (rows walked in chunks of CH, as T388's t388_chunked.py) around the frozen
walk_seg and rare.mass, called unmodified. It returns, per t0, the total (= estimator_t366.est up to summation order;
checked by `python frozen_est.py --check`) and the part from graded g12 nodes with g12 < 0.1 (the predicted
triple-tie share = tie / total). DEDUCTIVE: no random numbers."""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R1C = os.path.join(HERE, "frozen", "r1_cont")
SRC = os.path.join(HERE, "frozen", "src")
os.environ["T340_SRC"] = SRC
CF_SHA = "72fb793abe33a662a762d569e97c8bd2205f5817eb87156f1daf5096cfbd38d9"   # cheap_frontier/cf.py @620eb47b
for line in open(os.path.join(R1C, "t366_freeze_sha256.txt")):
    h, fn = line.split()
    p = os.path.join(SRC, "m16", "approaches", fn) if "/" in fn else os.path.join(R1C, fn)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, p
assert hashlib.sha256(open(os.path.join(SRC, "m16", "approaches", "cheap_frontier", "cf.py"), "rb").read()).hexdigest() == CF_SHA
sys.path.insert(0, R1C)
import estimator_t366 as E  # noqa: E402
import t366  # noqa: E402

assert E.DEDUCTIVE
FPE = 5948.0
CH = 40_000
TIE_G12 = 0.1


def graded_nodes(v2=0.0):
    return t366.graded_g12(v2)[0]


def est_split(model, t0s, tmax=None):
    rare, G, S, NT, TRI, NN = t366.rare, t366.G, t366.S, t366.NT, t366.TRI, t366.NN
    tmax = rare.TBIG if tmax is None else tmax
    model = tuple(np.asarray(w, np.float64) for w in model); t_start = min(t0s); flops = 0.0
    tot = {t0: 0.0 for t0 in t0s}; tie = {t0: 0.0 for t0 in t0s}
    for ri, tab in enumerate(G.TABLES):
        nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = t366.LINES[ri]; nrow = len(x)
        k = np.tile(TRI[:, 1], nrow // NT); k3 = np.tile(TRI[:, 2], nrow // NT); node_of_row = np.repeat(np.arange(len(nodes)), NN * NT)
        # g12 value of each row: rows are (v2 node, graded g12 node, triple); the graded nodes depend on v2 only beyond .4
        g12_row = np.concatenate([np.repeat(t366.graded_g12(v2)[0], NT) for v2 in nodes])
        is_tie = g12_row < TIE_G12
        wgt = tab["dens"] * tab["blind12"]
        acc = {t0: np.zeros(len(nodes)) for t0 in t0s}; acc_t = {t0: np.zeros(len(nodes)) for t0 in t0s}
        for a in range(0, nrow, CH):
            sl = slice(a, min(a + CH, nrow)); n = sl.stop - sl.start
            pieces, _, f = S.walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], t_start, tmax); flops += f
            for t0 in t0s:
                W, f = rare.mass(pieces, n, node_of_row[sl], nodes, t0); flops += f
                Wo = W * om[sl][:, None]
                acc[t0] += Wo.sum(0); acc_t[t0] += Wo[is_tie[sl]].sum(0)
        for t0 in t0s:
            tot[t0] += float(acc[t0] * wgt @ wts) / NT; tie[t0] += float(acc_t[t0] * wgt @ wts) / NT
            flops += nrow * len(nodes) * 3
    return tot, tie, flops


if __name__ == "__main__" and sys.argv[1:2] == ["--check"]:
    import json
    import time
    import models as MP
    m = MP.load(sys.argv[2], check_only=True); t0s = (1.0, 1.25, 1.5)
    t = time.time(); a, b, f = est_split(m, t0s); t1 = time.time()
    ref, fr = E.est(m, t0s); t2 = time.time()
    out = dict(model=sys.argv[2], rel={str(k): a[k] / ref[k] - 1 for k in t0s}, tie_share={str(k): b[k] / a[k] for k in t0s},
               C_split=f / FPE, C_ref=fr / FPE, sec_split=t1 - t, sec_ref=t2 - t1)
    print(out)
    json.dump(out, open(os.path.join(HERE, "results", f"check_{sys.argv[2]}.json"), "w"), indent=1)
