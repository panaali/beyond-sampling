"""T3 ceiling test: reduced system = clock-neuron gate itinerary S(x) + top-k values/positions, low inputs frozen
(V1/V2 as E27 ceiling2.py), clock neurons forced to S's masks. Bias = acc(reduced) - acc(true), paired.
Usage: python ceiling_gates.py [model names...]   (default: all 18). Writes results/<model>.json."""
import sys, json, os
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm
from models import all_models, clocks_data, clocks_weight

N, CH = 600_000, 50_000
KS, VS = (3, 4, 5), ("V1", "V2")


def run(M, x, force=None, masks=None):
    """force: bool (16,) neurons whose gates come from masks (N,10,16); returns logits and gate tensor."""
    Wi, Wh, Wo = M; h = np.zeros((len(x), 16)); B = np.empty((len(x), 10, 16), bool)
    for t in range(10):
        pre = h @ Wh.T + x[:, t:t+1] * Wi[None]; B[:, t] = pre > 0
        h = np.maximum(pre, 0) if force is None else np.where(force[None], pre * masks[:, t], np.maximum(pre, 0))
    return h @ Wo.T, B


def lows(x, o, k, nm):
    c = np.take_along_axis(x, o[:, k-1:k], 1)[:, 0]; n = 10 - k; j = np.arange(1, n + 1)
    if nm == "V1": return c, ndtri(ndtr(c)[:, None] * (n + 1 - j)[None] / (n + 1))
    return c, np.repeat((-norm.pdf(c) / ndtr(c))[:, None], n, 1)


def main(names):
    ms = all_models(); os.makedirs("results", exist_ok=True)
    for name in names:
        M = ms[name]; d5 = clocks_data(M, 5)
        rest = [j for j in range(16) if j not in d5]
        sets = {"D3": d5[:3], "D5": d5, "W3": clocks_weight(M, 3),
                "R3": sorted(np.random.default_rng(11).choice(rest, 3, replace=False).tolist()), "A16": list(range(16))}
        fm = {s: np.isin(np.arange(16), v) for s, v in sets.items()}
        rng = np.random.default_rng(7); rng_x = np.random.default_rng(13)
        okT = 0; diff = {}; flips = {}; itin = set()
        for _ in range(N // CH):
            x = rng.standard_normal((CH, 10)); o = np.argsort(-x, 1); tgt = o[:, 1]
            lg, B = run(M, x); ok = lg.argmax(1) == tgt; okT += ok.sum()
            itin.update((B[:, :, fm["D3"]].reshape(CH, -1) @ (1 << np.arange(30))).tolist())
            for k in KS:
                # X3 masks: same top-k, lows resampled iid N(0,1) truncated below c (inverse cdf)
                c = np.take_along_axis(x, o[:, k-1:k], 1)[:, 0]
                xs = x.copy(); u = rng_x.random((CH, 10 - k)) * ndtr(c)[:, None]
                np.put_along_axis(xs, o[:, k:], ndtri(np.maximum(u, 1e-300)), 1)
                _, Bs = run(M, xs)
                for nm in VS:
                    _, v = lows(x, o, k, nm); xr = x.copy(); np.put_along_axis(xr, o[:, k:], v, 1)
                    for s in ("none", "D3", "D5", "W3", "R3", "X3", "A16"):
                        if s == "none": lr, _ = run(M, xr)
                        elif s == "X3": lr, _ = run(M, xr, fm["D3"], Bs)
                        else: lr, _ = run(M, xr, fm[s], B)
                        okr = lr.argmax(1) == tgt; key = f"{s}_k{k}_{nm}"
                        diff[key] = diff.get(key, 0) + int(okr.sum() - ok.sum())
                        flips[key] = flips.get(key, 0) + int((okr != ok).sum())
        out = {"model": name, "N": N, "acc_true": okT / N, "sets": sets, "n_itin_D3": len(itin),
               "bias_pp": {kk: 100 * v / N for kk, v in diff.items()},
               "se_pp": {kk: 100 * np.sqrt(flips[kk]) / N for kk in diff}}
        json.dump(out, open(f"results/{name}.json", "w"), indent=1)
        b = out["bias_pp"]
        print(f"{name:10s} acc {out['acc_true']:.4f} itin {len(itin)} | k4V1 " +
              " ".join(f"{s} {b[f'{s}_k4_V1']:+.3f}" for s in ("none", "D3", "D5", "W3", "R3", "X3", "A16")), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(all_models()))
