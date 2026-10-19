"""T407: the frozen graded-g12 estimator (claude/t218-tie-tail-labels @5b4a1250, sha256 asserted) on M_16,10's rating grid,
plus the deterministic trust flag = the estimator's own share of event mass from g12 nodes < .1. The wrapper repeats
t366.est_e1's loop with the same frozen functions; its estimates are checked against estimator_t366.est.
-> results/t407_est.json"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S0 = "/tmp/claude-0/-home-user-clean-neurons/18e70fa5-7ff5-5fac-af53-4fea2351128f/scratchpad"
R1C = os.path.join(S0, "r1c366", "m16", "approaches", "r1_cont")
SHA = {"estimator_t366.py": "6bd6822e51bd06e807dc7c51baefee04931ad49a86f168c7ed6136473350dffe",
       "t366.py": "997a1b60052757b3048c9baf846245654e8bdc6d2c75f6e5224a746b18064774",
       "t340c.py": "7e6a749d67284d4e634bfb302087b9559da4956a6d5e6a1d29dd381959ec9236",
       "t354.py": "abed36c8db9dbecfe36b6e5ab41090363778a8d1485f0e7599e6f23298232dde",
       "walkfix.py": "8e21b6a11c7d9151d2f02f84d1e09f5225b1e605501aa0e57aeb398a1d432d7e"}
for f, h in SHA.items():
    assert hashlib.sha256(open(os.path.join(R1C, f), "rb").read()).hexdigest() == h, f
os.environ["T340_SRC"] = os.path.join(S0, "src620")
sys.path.insert(0, R1C)
import estimator_t366 as E  # noqa: E402
import t366 as T  # noqa: E402

T1 = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


def est_with_flag(model, t0s, tmax=T.rare.TBIG):
    model = tuple(np.asarray(w, np.float64) for w in model); res = {t: 0.0 for t in t0s}; low = {t: 0.0 for t in t0s}
    flops = 0.0; t_start = min(t0s)
    for ri, tab in enumerate(T.G.TABLES):
        nodes, wts = tab["nodes"], tab["wts"]; x, dx, om = T.LINES[ri]; nrow = len(x)
        k = np.tile(T.TRI[:, 1], nrow // T.NT); k3 = np.tile(T.TRI[:, 2], nrow // T.NT)
        node_of_row = np.repeat(np.arange(len(nodes)), T.NN * T.NT)
        g12_row = np.concatenate([np.repeat(T.graded_g12(v2)[0], T.NT) for v2 in nodes])      # same order as LINES
        is_low = (g12_row < 0.1).astype(float)
        wgt = tab["dens"] * tab["blind12"]
        step = 4 * T.NN * T.NT                          # T407: 4 v2 nodes per chunk (memory limit); walks are per row
        for a in range(0, nrow, step):
            sl = slice(a, min(a + step, nrow)); n = sl.stop - sl.start
            pieces, _, f = T.S.walk_seg(model, x[sl], dx[sl], k[sl], k3[sl], t_start, tmax); flops += f
            for t0 in t0s:
                W, f = T.rare.mass(pieces, n, node_of_row[sl], nodes, t0); flops += f
                res[t0] += float((W * om[sl, None]).sum(0) * wgt @ wts) / T.NT
                low[t0] += float((W * (om[sl] * is_low[sl])[:, None]).sum(0) * wgt @ wts) / T.NT
                flops += n * len(nodes) * 3
    return res, {t: (low[t] / res[t] if res[t] > 0 else None) for t in t0s}, flops


m = T.TC.cf.lb3_models()["orig"]
t = time.time(); r, share, f = est_with_flag(m, T1); sec = time.time() - t
print("wrapper", {k: f"{v:.4e}" for k, v in r.items()}, {k: round(v, 3) for k, v in share.items()}, f"C {f / 5948:.0f}", f"{sec:.0f}s", flush=True)
# equality check (declared deviation): estimator_t366.est does not fit the 2.5 GB limit, so the chunked wrapper is checked
# against the T218 child's own stored dev run of estimator_t366 on orig (results_t366_dev.json, t0 = 1.75, 2.0)
ref = json.load(open(os.path.join(R1C, "results_t366_dev.json")))["orig"]["est"]
rc, sh2, fc = est_with_flag(m, (1.75, 2.0))
rel = max(abs(rc[float(k)] - v) / abs(v) for k, v in ref.items())
print("chunked wrapper vs stored estimator_t366 (orig, 1.75 / 2.0): max rel diff", rel, flush=True)
json.dump(dict(sha256=SHA, commit="5b4a1250", est={str(k): v for k, v in r.items()}, check_vs_stored=dict(ref=ref, wrapper={str(k): v for k, v in rc.items()}, max_rel_diff=rel),
               max_rel_diff=rel, flag_share_g12_lt_0p1={str(k): v for k, v in share.items()}, C=f / 5948,
               note="chunked by 4 v2 nodes (memory); C as executed is >= the unchunked prefix-tree count"),
          open(os.path.join(HERE, "results", "t407_est.json"), "w"), indent=1)
