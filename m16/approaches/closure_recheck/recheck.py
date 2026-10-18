"""T303: AGENTS rule-5 re-check of T298's post hoc closure "top-4 exact + spread lows", rebuilt from its README
definition (not from T298's code). Definition used:
  keep the 4 largest inputs (positions and values) exact; the other 6 positions, taken in time order, receive the
  values Q(Phi(v4) * (7-j)/7), j = 1..6, once descending and once ascending in time; the closure verdict is the
  average of the two points' correctness (network output == position of the true 2nd largest).
Bias b = mean over x ~ N(0, I) of (verdict - network correct on x); regions as T292. Seeds 303000+ (disjoint from
T298's 298000+/298500+). Also re-measures R0 (top-3 exact, 7 lows at -phi(v3)/Phi(v3)) as a reference.
usage: python recheck.py dev2|lb3 [N]"""
import json
import sys
import numpy as np
from scipy.stats import norm

panel = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000


def net_argmax(W, X):
    a, B, C = W
    s = np.zeros((X.shape[0], B.shape[0]))
    for col in X.T:                                   # iterate over time steps
        s = np.maximum(s @ B.T + np.outer(col, a), 0)
    return np.argmax(s @ C.T, axis=1)


def verdicts(W, X):
    rank = np.argsort(X, axis=1)[:, ::-1]             # positions sorted by value, descending
    srt = np.sort(X, axis=1)[:, ::-1]
    answer = rank[:, 1]
    ok_true = net_argmax(W, X) == answer
    n = X.shape[0]
    rows = np.arange(n)[:, None]
    # top-4 + spread lows
    lows_pos = np.sort(rank[:, 4:], axis=1)           # 6 free positions in time order
    q = np.arange(6, 0, -1) / 7.0                     # (7-j)/7 for j = 1..6, descending
    vals = norm.ppf(norm.cdf(srt[:, 3:4]) * q[None, :])
    P1, P2 = X.copy(), X.copy()
    P1[rows, lows_pos] = vals
    P2[rows, lows_pos] = vals[:, ::-1]
    v_t4s = 0.5 * ((net_argmax(W, P1) == answer).astype(float) + (net_argmax(W, P2) == answer))
    # R0 reference
    mu = -norm.pdf(srt[:, 2]) / norm.cdf(srt[:, 2])
    P0 = X.copy()
    P0[rows, rank[:, 3:]] = mu[:, None]
    v_r0 = (net_argmax(W, P0) == answer).astype(float)
    g12, g23 = srt[:, 0] - srt[:, 1], srt[:, 1] - srt[:, 2]
    region = np.select([srt[:, 1] <= 0, np.minimum(g12, g23) >= 0.05, g23 <= g12], [0, 3, 1], 2)
    return ok_true.astype(float), {"top4_spread": v_t4s, "R0": v_r0}, region


if panel == "dev2":
    Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
    models = [(str(nm), (Z["Wi"][i].astype(float), Z["Wh"][i].astype(float), Z["Wo"][i].astype(float)))
              for i, nm in enumerate(Z["names"])]
    truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
else:
    sys.path.insert(0, "/home/user/lb3/m16")
    import leaderboard as L
    models = list(L.models().items())
    truth = {k: v["acc"] for k, v in json.load(open("/home/user/lb3/" + L.truths_path(16, 10))).items()}
out = {}
for i, (name, W) in enumerate(models):
    g = np.random.default_rng(303000 + (0 if panel == "dev2" else 1000) + i)
    acc = {}
    for _ in range(N // 100_000):
        X = g.standard_normal((100_000, 10))
        ok, V, reg = verdicts(W, X)
        for cn, v in V.items():
            d = v - ok
            a = acc.setdefault(cn, np.zeros(5))
            a[0] += d.sum()
            for r in range(4):
                a[1 + r] += d[reg == r].sum()
    out[name] = {"truth": truth[name], **{cn: (100 * a / N).tolist() for cn, a in acc.items()}}
    print(name, {cn: round(v[0], 3) for cn, v in out[name].items() if cn != "truth"}, flush=True)
json.dump(out, open(f"results/{panel}.json", "w"), indent=1)
names = list(out)
acc = np.array([100 * out[n]["truth"] for n in names])
for cn in ("top4_spread", "R0"):
    B = np.array([out[n][cn] for n in names])
    b = B[:, 0]
    print(f"{panel} {cn}: RMS {np.sqrt((b ** 2).mean()):.3f} mean {b.mean():+.3f} slope {np.polyfit(acc, b, 1)[0]:+.4f} "
          f"max {np.abs(b).max():.2f} | regions mean/RMS " +
          " ".join(f"{nm} {B[:, 1 + r].mean():+.3f}/{np.sqrt((B[:, 1 + r] ** 2).mean()):.3f}"
                   for r, nm in enumerate(("blind", "b23", "b12", "tail"))))
