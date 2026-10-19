"""T405 truth: central finite differences with common random numbers under the exact conditional law of x given
g23 > t0 (sampler.py, own code). Each frozen direction u is probed at the ladder delta_u x LADDER on the same draws;
the FD used for scoring is the one at the largest level whose realised changes satisfy |P+/P - 1| <= .5 and
|P-/P - 1| <= .5 (the smallest level if none does) - T372's amendment A1 rule. Resolved: |FD| > 3 s.e.
usage: python truth.py <model> <t0> <N> -> results/truth_<model>_<t0>.json (checkpointed, resumable by extending N
only through a fresh run with the same seed: draws are a deterministic stream of fixed-size chunks)"""
import json
import os
import sys
import time

import numpy as np

import make_cases
import sampler as S

HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = np.array([1.0, 1 / 3, 1 / 10, 1 / 30, 1 / 100])
SH = [(16,), (16, 16), (10, 16)]
CHUNK = 4096


def unflat(v):
    out, i = [], 0
    for s in SH:
        n = int(np.prod(s)); out.append(v[i:i + n].reshape(s)); i += n
    return tuple(out)


def summarize(D, pg, n, h0, hp, hm, flips, dl):
    q = h0 / n; P = pg * q; md = (hp - hm) / n; vd = flips / n - md ** 2
    fd = pg * md / (2 * dl); se = pg * np.sqrt(vd / n) / (2 * dl); Pp, Pm = pg * hp / n, pg * hm / n
    ok = (np.abs(Pp / P - 1) <= .5) & (np.abs(Pm / P - 1) <= .5) if P > 0 else np.zeros(dl.shape, bool)
    pick = [int(np.flatnonzero(ok[i])[0]) if ok[i].any() else len(LADDER) - 1 for i in range(len(dl))]
    r = np.arange(len(dl))
    return dict(model=D["model"], t0=D["t0"], n=n, p_gap=pg, hits0=h0, P=P, P_se=pg * np.sqrt(q * (1 - q) / n), ladder=LADDER.tolist(),
                delta=dl.tolist(), hits_plus=hp.tolist(), hits_minus=hm.tolist(), flips=flips.tolist(), P_plus=Pp.tolist(),
                P_minus=Pm.tolist(), FD=fd.tolist(), FD_se=se.tolist(), level=pick, FD_sel=fd[r, pick].tolist(),
                FD_sel_se=se[r, pick].tolist())


def main(nm, t0, N):
    D = json.load(open(make_cases.path(nm, t0))); ci = D["case"]
    w0 = np.array(D["w"]); U = np.array(D["dirs"]); dl = np.array(D["delta"])[:, None] * LADDER[None]
    ms = [unflat(w0)] + [unflat(w0 + s * dl[i, l] * U[i]) for i in range(len(U)) for l in range(len(LADDER)) for s in (1, -1)]
    St = S.stack(ms); pg, cdf = S.v2_table(t0); rng = np.random.default_rng(405100 + ci)
    n = 0; h0 = 0; hp = np.zeros(dl.shape, int); hm = np.zeros(dl.shape, int); fl = np.zeros(dl.shape, int); t = time.time()
    out_path = os.path.join(HERE, "results", f"truth_{nm}_{t0}.json"); k = 0
    while n < N:
        x, p3 = S.draw(rng, CHUNK, t0, cdf); H = S.argmax_stack(St, x) == p3[None]
        h0 += int(H[0].sum()); Hp = H[1::2].reshape(dl.shape + (CHUNK,)); Hm = H[2::2].reshape(dl.shape + (CHUNK,))
        hp += Hp.sum(2); hm += Hm.sum(2); fl += (Hp != Hm).sum(2); n += CHUNK; k += 1
        if k % 500 == 0 or n >= N:
            out = summarize(D, pg, n, h0, hp, hm, fl, dl); out["sec"] = time.time() - t; out["forward_passes"] = n * len(ms)
            json.dump(out, open(out_path, "w"), indent=1)
            print(nm, t0, n, "P %.4e" % out["P"], "lvl", out["level"], "FD/P", np.round(np.array(out["FD_sel"]) / max(out["P"], 1e-300), 2),
                  "z", np.round(np.array(out["FD_sel"]) / np.maximum(np.array(out["FD_sel_se"]), 1e-300), 1), "%.0fs" % out["sec"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(float(sys.argv[3])))
