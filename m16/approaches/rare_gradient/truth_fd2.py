"""T372 truth, amendment A1 (README): as truth_fd.py, but each frozen direction is probed at a ladder of steps
delta_u x LADDER on the same draws (common random numbers across levels and directions). The reported FD per direction
is the one at the largest level whose realised changes satisfy |P+/P - 1| <= .5 and |P-/P - 1| <= .5 (the smallest
level if none does). usage: python truth_fd2.py model N -> results/truth_fd2_<model>.json (checkpointed)"""
import json
import os
import sys
import time

import numpy as np

import rnn
import truthc_copy as TCC

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = {"orig": 372110, "dev3_s526": 372111, "dev3_s528": 372112}
LADDER = np.array([1.0, 1 / 3, 1 / 10, 1 / 30, 1 / 100])


def hits(model, x, p3):
    return rnn.forward(model, x).argmax(1) == p3


def summarize(nm, t0, pg, n, h0, hp, hm, dd, dl):
    q = h0 / n; P = pg * q; mean_d = (hp - hm) / n; var_d = dd / n - mean_d ** 2
    fd = pg * mean_d / (2 * dl); se = pg * np.sqrt(var_d / n) / (2 * dl)
    Pp, Pm = pg * hp / n, pg * hm / n
    ok = (np.abs(Pp / P - 1) <= .5) & (np.abs(Pm / P - 1) <= .5)
    pick = [int(np.flatnonzero(ok[i])[0]) if ok[i].any() else len(LADDER) - 1 for i in range(len(dl))]
    r = np.arange(len(dl))
    return dict(model=nm, t0=t0, n=n, p_gap=pg, hits0=h0, P=P, P_se=pg * np.sqrt(q * (1 - q) / n), ladder=LADDER.tolist(),
                delta=dl.tolist(), hits_plus=hp.tolist(), hits_minus=hm.tolist(), flips=dd.tolist(), P_plus=Pp.tolist(),
                P_minus=Pm.tolist(), FD=fd.tolist(), FD_se=se.tolist(), level=pick, FD_sel=fd[r, pick].tolist(), FD_sel_se=se[r, pick].tolist())


def main(nm, N, chunk=250_000):
    D = json.load(open(os.path.join(HERE, "results", f"dirs_{nm}.json"))); t0 = D["t0"]
    w0 = np.array(D["w"]); sh = [(16,), (16, 16), (10, 16)]; U = np.array(D["dirs"]); dl = np.array(D["delta"])[:, None] * LADDER[None]
    base = rnn.unflat(w0, sh)
    plus = [[rnn.unflat(w0 + dl[i, l] * U[i], sh) for l in range(len(LADDER))] for i in range(len(U))]
    minus = [[rnn.unflat(w0 - dl[i, l] * U[i], sh) for l in range(len(LADDER))] for i in range(len(U))]
    draw = TCC.sampler(t0); rng = np.random.default_rng(SEED[nm]); pg = TCC.p_gap(t0)
    n = 0; h0 = 0; hp = np.zeros(dl.shape, int); hm = np.zeros(dl.shape, int); dd = np.zeros(dl.shape); t = time.time()
    path = os.path.join(HERE, "results", f"truth_fd2_{nm}.json")
    while n < N:
        m = min(chunk, N - n); x, p3 = draw(rng, m); h0 += int(hits(base, x, p3).sum())
        for i in range(len(U)):
            for l in range(len(LADDER)):
                a = hits(plus[i][l], x, p3); c = hits(minus[i][l], x, p3)
                hp[i, l] += int(a.sum()); hm[i, l] += int(c.sum()); dd[i, l] += float((a != c).sum())
        n += m
        if n % (20 * chunk) == 0 or n == N:
            out = summarize(nm, t0, pg, n, h0, hp, hm, dd, dl); out["sec"] = time.time() - t
            out["forward_passes"] = n * (1 + 2 * dl.size); json.dump(out, open(path, "w"), indent=1)
            print(nm, n, "P %.4e" % out["P"], "level", out["level"], "FD/P", np.round(np.array(out["FD_sel"]) / out["P"], 3),
                  "z", np.round(np.array(out["FD_sel"]) / np.array(out["FD_sel_se"]), 1), "%.0fs" % out["sec"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(float(sys.argv[2])))
