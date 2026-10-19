"""T425 verification (SAMPLING truths; own forward and own conditional sampler, not core.py/walk.py; new seeds).
Run once on a frozen edit (results/<tag>/edit.json, sha256 checked):
  python verify.py <tag>  -> results/<tag>/verify.json
E1(t0): exact conditional law given g23 > t0, base / edit / 10 random edits (same S, same L2) on the same draws;
accuracy: plain MC on N(0, I_T), base / edit (1e8) and random edits (first 1e7), paired."""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr, ndtri

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import memo  # noqa: E402  (disk memo only; no sampler/forward code from core)
G2 = np.linspace(-8.0, 9.0, 1_700_001)
N_R_EDITS = 10; N_CAP = int(float(os.environ.get("T425_NCAP", 3e8)))      # env overrides: smoke tests only
N_ACC = int(float(os.environ.get("T425_NACC", 1e8))); N_ACC_R = min(N_ACC, 10_000_000)


def mk(w, H, T):
    w = np.asarray(w, float); return w[:H], w[H:H + H * H].reshape(H, H), w[H + H * H:].reshape(T, H)


def pick(models, x):
    """argmax of several models on the same inputs (float64 forward, own code)."""
    out = []
    for Wi, Wh, Wo in models:
        h = np.zeros((len(x), len(Wi))); A = Wh.T.copy()
        for t in range(x.shape[1]):
            h = h @ A; h += np.outer(x[:, t], Wi); np.maximum(h, 0, out=h)
        out.append((h @ Wo.T).argmax(1))
    return np.array(out)


class Cond:
    """x | g23 > t0: v2 from its marginal T(T-1) phi(v)(1-Phi(v)) Phi(v-t0)^(T-2) (inverse CDF on G2), v1 > v2,
    the T-2 others iid N truncated below v2 - t0 (v3 = their max), positions uniform."""
    def __init__(self, T, t0):
        lf = np.log(T * (T - 1)) - G2 ** 2 / 2 - .5 * np.log(2 * np.pi) + np.log(np.maximum(ndtr(-G2), 1e-300)) + (T - 2) * np.log(np.maximum(ndtr(G2 - t0), 1e-300))
        f = np.exp(lf); c = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(G2))])
        self.pg = c[-1]; self.c = c / c[-1]; self.T, self.t0 = T, t0

    def draw(self, rng, n):
        T = self.T; v2 = np.interp(rng.random(n), self.c, G2); v1 = -ndtri(rng.random(n) * ndtr(-v2))
        rest = ndtri(ndtr(v2 - self.t0)[:, None] * rng.random((n, T - 2)))      # iid below v2 - t0
        vals = np.column_stack([v1, v2, rest]); pos = np.argsort(rng.random((n, T)), 1)
        x = np.empty((n, T)); x[np.arange(n)[:, None], pos] = vals
        return x


def ratio_se(hb, he, both, n):
    if not (hb and he):
        return float("nan")
    vb, ve, vbe = hb / n, he / n, both / n; r = he / hb
    return r * np.sqrt(max((1 - vb) / vb / n + (1 - ve) / ve / n - 2 * (vbe - vb * ve) / (vb * ve) / n, 0))


def main(tag):
    out = os.path.join(HERE, "results", tag); E = json.load(open(os.path.join(out, "edit.json")))
    assert hashlib.sha256(json.dumps(E["w_edit"]).encode()).hexdigest() == E["sha256_w_edit"], "edit.json changed"
    H, T, t0 = E["H"], E["T"], E["t0"]; wb = np.array(E["w_base"]); we = np.array(E["w_edit"]); S = np.array(E["S"])
    eps = float(np.linalg.norm(we - wb)); sd = 425_900_000 + H * 1000 + T * 10 + (5 if "zoo" in tag else 0) + int(tag[-1])
    rng = np.random.default_rng(sd); R = []
    for _ in range(N_R_EDITS):
        u = rng.standard_normal(len(S)); v = wb.copy(); v[S] += eps * u / np.linalg.norm(u); R.append(v)
    models = [mk(wb, H, T), mk(we, H, T)] + [mk(v, H, T) for v in R]
    C = Cond(T, t0); q = E["P0_design"] / C.pg                                               # design base P0 / p_gap
    BLK = min(int(1e7), N_CAP)                                     # memoised blocks (resumable); N rounded up to whole blocks
    N_E = int(min(N_CAP, np.ceil(1000 / q / BLK) * BLK)); N_R = int(min(N_E, np.ceil(300 / q / BLK) * BLK))
    t = time.time(); ch = 200_000; key = f"{tag}/verify_{E['sha256_w_edit'][:12]}"

    def e1_block(b):
        rng = np.random.default_rng([sd + 1, b]); n = 0; L = len(models) if b * BLK < N_R else 2
        hits = np.zeros(L); hits_r = np.zeros(L); both_r = np.zeros(L); ranks = np.zeros((2, T), int)
        while n < BLK:
            m = min(ch, BLK - n); x = C.draw(rng, m); rk = np.argsort(np.argsort(-x, 1), 1)      # rk[i, pos] = rank (0 = largest)
            P = pick(models[:L], x); r3 = np.argsort(-x, 1)[:, 2]; hit = P == r3[None]
            hits += hit.sum(1); both_r += (hit & hit[0]).sum(1)
            for j in (0, 1):
                np.add.at(ranks[j], rk[np.arange(m), P[j]], 1)
            n += m
        print(tag, "E1 block", b, hits[:2], "%.0fs" % (time.time() - t), flush=True)
        return hits, both_r, ranks

    nbE = N_E // BLK; hits = np.zeros(len(models)); both = 0; hits_r = np.zeros(len(models)); both_r = np.zeros(len(models)); ranks = np.zeros((2, T), int)
    for b in range(nbE):
        h, br, rk = memo(f"{key}_E{b}", lambda b=b: e1_block(b)); hits[:2] += h[:2]; both += br[1]; ranks += rk
        if len(h) == len(models):
            hits_r += h; both_r += br
    rE = hits[1] / hits[0]; rEse = ratio_se(hits[0], hits[1], both, N_E)
    rr = [dict(ratio=float(hits_r[i] / hits_r[0]) if hits_r[0] else None, ratio_se=ratio_se(hits_r[0], hits_r[i], both_r[i], N_R)) for i in range(2, len(models))]

    def acc_block(b):                                              # accuracy (paired); random edits in block 0 (first 1e7)
        rng = np.random.default_rng([sd + 2, b]); n = 0; L = len(models) if b == 0 else 2; ok = np.zeros(L); d1 = np.zeros(L); d2 = np.zeros(L)
        while n < BLK:
            m = min(ch, BLK - n); x = rng.standard_normal((m, T)); tgt = np.argsort(-x, 1)[:, 1]
            c = (pick(models[:L], x) == tgt[None]).astype(float); ok += c.sum(1); dd = c - c[0]; d1 += dd.sum(1); d2 += (dd ** 2).sum(1); n += m
        print(tag, "acc block", b, "%.0fs" % (time.time() - t), flush=True)
        return ok, d1, d2

    ok = np.zeros(len(models)); d1 = np.zeros(len(models)); d2 = np.zeros(len(models)); nr = BLK
    for b in range(N_ACC // BLK):
        o, x1, x2 = memo(f"{key}_A{b}", lambda b=b: acc_block(b)); k = len(o); ok[:k] += o; d1[:k] += x1; d2[:k] += x2
    dacc = d1[1] / N_ACC; dacc_se = np.sqrt(max(d2[1] / N_ACC - dacc ** 2, 0) / N_ACC)
    for i, r in enumerate(rr):
        mu = d1[2 + i] / nr; r.update(dacc=float(mu), dacc_se=float(np.sqrt(max(d2[2 + i] / nr - mu ** 2, 0) / nr)))
        r["holds"] = bool(r["ratio"] is not None and r["ratio"] <= .1 and r["dacc"] >= -5e-4)
    a_ok = bool(rE <= .1 and rE + 1.96 * rEse <= .15); b_ok = bool(dacc >= -5e-4)
    rmin = min(r["ratio"] for r in rr if r["ratio"] is not None); c_ok = bool(not any(r["holds"] for r in rr) and rE < rmin)
    pg = C.pg; conv = dict(rank_counts_base=ranks[0].tolist(), rank_counts_edit=ranks[1].tolist(),
                           P_largest_base=pg * ranks[0][0] / N_E, P_largest_edit=pg * ranks[1][0] / N_E,
                           P_anywrong_base=pg * (N_E - ranks[0][1]) / N_E, P_anywrong_edit=pg * (N_E - ranks[1][1]) / N_E)
    conv["d_largest"] = conv["P_largest_edit"] - conv["P_largest_base"]; conv["d_E1"] = pg * (hits[1] - hits[0]) / N_E
    conv["largest_rel"] = ranks[1][0] / ranks[0][0] - 1 if ranks[0][0] else None
    res = dict(tag=tag, H=H, T=T, t0=t0, K=E["K"], eps=eps, design_status=E["status"], N_E=N_E, N_R=N_R, p_gap=pg,
               P_base=pg * hits[0] / N_E, P_edit=pg * hits[1] / N_E, hits_base=int(hits[0]), hits_edit=int(hits[1]), ratio=rE, ratio_se=rEse,
               acc_base=ok[0] / N_ACC, dacc=dacc, dacc_se=dacc_se, random=rr, random_ratio_min=rmin,
               random_ratio_median=float(np.median([r["ratio"] for r in rr if r["ratio"] is not None])),
               random_dacc_median=float(np.median([r["dacc"] for r in rr])), conversion=conv,
               holds=dict(a_ratio=a_ok, b_acc=b_ok, c_random=c_ok, all=a_ok and b_ok and c_ok),
               fwd=float(N_E * 2 + N_R * N_R_EDITS + N_ACC * 2 + BLK * N_R_EDITS), sec=time.time() - t)
    json.dump(res, open(os.path.join(out, "verify.json"), "w"), indent=1, default=float)
    print(tag, "ratio %.3f +- %.3f dacc %.4f pp holds %s" % (rE, rEse, 100 * dacc, res["holds"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
