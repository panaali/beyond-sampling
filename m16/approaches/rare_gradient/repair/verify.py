"""T402 verification (independent of the design: different sampler/forward code path, new seeds; run once on the
frozen edit in edit.json, whose sha256 is checked). SAMPLING (truths).
  python verify.py e1 t0 N seed      -> results/ver_e1_<t0>.json  conditional-law MC (truthc sampler), base & edit on the same draws
  python verify.py e2 t N seed       -> results/ver_e2_<t>.json   conditional-law MC of E2(t) = {wrong, g12 >= t, g23 >= t}
  python verify.py acc N seed        -> results/ver_acc.json      plain MC accuracy, base & edit on the same inputs
Each reports both P's, the ratio, and paired s.e. (the paired difference d = 1_edit - 1_base per draw)."""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import truthc_copy as TCC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")


def load():
    E = json.load(open(os.path.join(HERE, "edit.json")))
    h = hashlib.sha256(json.dumps(E["w_edit"]).encode()).hexdigest(); assert h == E["sha256_w_edit"], "edit.json changed"
    def mk(w):
        w = np.asarray(w, float); return w[:16], w[16:272].reshape(16, 16), w[272:].reshape(10, 16)
    return mk(E["w_base"]), mk(E["w_edit"])


def pick(model, x):                                  # own forward (not rnn.py)
    Wi, Wh, Wo = model; h = np.zeros((len(x), 16)); WhT = Wh.T.copy()
    for t in range(10):
        h = h @ WhT; h += x[:, t:t + 1] * Wi; np.maximum(h, 0, out=h)
    return (h @ Wo.T).argmax(1)


def e2_sampler(t):
    G = TCC.GRID; f = 90 * np.exp(-G ** 2 / 2) / np.sqrt(2 * np.pi) * ndtr(-(G + t)) * ndtr(G - t) ** 8
    c = np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(G))]); pg = c[-1]; c = c / c[-1]
    def draw(rng, n):
        v2 = np.interp(rng.random(n), c, G); v1 = -ndtri(rng.random(n) * ndtr(-(v2 + t)))
        v3 = ndtri(ndtr(v2 - t) * rng.random(n) ** (1 / 8)); lows = ndtri(ndtr(v3)[:, None] * rng.random((n, 7)))
        vals = np.column_stack([v1, v2, v3, lows]); perm = np.argsort(rng.random((n, 10)), 1)
        x = np.empty_like(vals); x[np.arange(n)[:, None], perm] = vals
        return x, perm
    return draw, pg


def run(kind, arg, N, seed, chunk=250_000):
    base, ed = load(); rng = np.random.default_rng(seed); n = 0; hb = he = 0; s1 = s2 = 0.0; t = time.time()
    split = np.zeros((2, 10), int)                     # hits by picked rank (base, edit), for E1-type events
    if kind == "e1":
        draw = TCC.sampler(arg); scale = TCC.p_gap(arg)
    elif kind == "e2":
        draw, scale = e2_sampler(arg)
    else:
        scale = 1.0
    name = f"ver_{kind}_{arg}.json" if kind != "acc" else "ver_acc.json"
    while n < N:
        m = min(chunk, N - n)
        if kind == "e1":
            x, p3 = draw(rng, m); pb, pe = pick(base, x), pick(ed, x); b, e = pb == p3, pe == p3
            perm = np.argsort(-x, 1); rank_b = (perm == pb[:, None]).argmax(1); rank_e = (perm == pe[:, None]).argmax(1)
            np.add.at(split[0], rank_b, 1); np.add.at(split[1], rank_e, 1)
        elif kind == "e2":
            x, perm = draw(rng, m); p2 = perm[:, 1]; b, e = pick(base, x) != p2, pick(ed, x) != p2
        else:
            x = rng.standard_normal((m, 10)); tgt = np.argsort(-x, 1)[:, 1]; b, e = pick(base, x) == tgt, pick(ed, x) == tgt
        hb += int(b.sum()); he += int(e.sum()); dd = e.astype(float) - b; s1 += dd.sum(); s2 += (dd ** 2).sum(); n += m
        if n % (40 * chunk) == 0 or n == N:
            mu = s1 / n; se = np.sqrt(max(s2 / n - mu ** 2, 0) / n); Pb, Pe = scale * hb / n, scale * he / n
            # delta-method s.e. of the ratio from the paired counts
            r = he / hb if hb else np.nan
            vb, ve = hb / n, he / n; cov = (vb + ve - (s2 / n)) / 2 - vb * ve           # E[b e] - E[b]E[e]
            rse = r * np.sqrt(max((vb - vb ** 2) / vb ** 2 + (ve - ve ** 2) / ve ** 2 - 2 * cov / (vb * ve), 0) / n) if hb and he else np.nan
            out = dict(kind=kind, arg=arg, n=n, seed=seed, hits_base=hb, hits_edit=he, P_base=Pb, P_edit=Pe, ratio=r, ratio_se=rse,
                       diff=scale * mu, diff_se=scale * se, sec=time.time() - t)
            if kind == "e1":
                out["picked_rank_counts"] = dict(base=split[0].tolist(), edit=split[1].tolist())
            json.dump(out, open(os.path.join(RES, name), "w"), indent=1)
            print(kind, arg, n, "P_base %.4e P_edit %.4e ratio %.4f +- %.4f diff %.3e +- %.1e" % (Pb, Pe, r, rse, scale * mu, scale * se), flush=True)


if __name__ == "__main__":
    k = sys.argv[1]
    if k == "acc":
        run("acc", 0, int(float(sys.argv[2])), int(sys.argv[3]))
    else:
        run(k, float(sys.argv[2]), int(float(sys.argv[3])), int(sys.argv[4]))
