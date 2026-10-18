"""T308 per-class anchor quantities and the 10 universal summaries for the 80 DEV2 models and the 5 LB3 seeds.
Imports T300's tb/r0 unchanged (read-only worktree of claude/t218-tie-tail-labels). Diagnostic."""
import json
import sys
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

sys.path.insert(0, "/home/user/t218/m16/approaches/r0_integral")
import r0  # noqa: E402
import tb  # noqa: E402

TRI = r0.TRIPLES
n = len(TRI)
rr = np.arange(n)
V2 = tb.MF["q"](0.5)
V1 = V2 + tb.MF["g12"]
V3M = V2 - tb.MF["g23"]
lam = norm.pdf(V3M) / ndtr(V3M)
SIG_LOW = float(np.sqrt(1 - V3M * lam - lam ** 2))      # sd of N(0,1) | X < v3 at the median anchor


def margin_and_grad_lows(model):
    x = r0.r0_inputs(TRI, np.full(n, V1), np.full(n, V2), np.full(n, V3M))
    L = r0.logits(model, x)
    L2 = L.copy(); L2[rr, TRI[:, 1]] = -np.inf; j = L2.argmax(1)
    m = L[rr, TRI[:, 1]] - L[rr, j]
    free = np.array([[p for p in range(10) if p not in t] for t in TRI])
    G = np.empty((n, 7))
    for k in range(7):
        dx = np.zeros((n, 10)); dx[rr, free[:, k]] = 1.0
        _, dL = r0.logits(model, x, dx)
        G[:, k] = dL[rr, TRI[:, 1]] - dL[rr, j]
    # active-gate count at the median anchor
    Wi, Wh, Wo = model
    h = np.zeros((n, Wh.shape[0])); act = np.zeros(n)
    for t in range(10):
        z = h @ Wh.T + x[:, t:t + 1] * Wi[None]; act += (z > 0).sum(1); h = np.maximum(z, 0)
    return m, np.linalg.norm(G, axis=1), act


def stored(path):
    rx, pr = {}, {}
    for line in open(path):
        d = json.loads(line)
        if d["fam"] == "rx" and d["cfg"] == {"Q": 8, "stride": 1, "kgroups": 3}:
            rx[d["model"]] = d["est"]
        if d["fam"] == "probe" and d["cfg"] == {"version": 1, "iters": 8, "stride": 1}:
            pr[d["model"]] = d["est"]
    return rx, pr


Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
models = [(str(nm), (Z["Wi"][i].astype(float), Z["Wh"][i].astype(float), Z["Wo"][i].astype(float)), "dev2")
          for i, nm in enumerate(Z["names"])]
truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as LB  # noqa: E402
lbm = LB.models()
lbt = json.load(open("/home/user/lb3/" + LB.truths_path(16, 10)))
for s in range(5):
    models.append((f"seed{s}", lbm[f"seed{s}"], "lb3"))
    truth[f"seed{s}"] = lbt[f"seed{s}"]["acc"]
rx_d, pr_d = stored("/home/user/dl/t286/sweep_raw_dev2.jsonl")
rx_l, pr_l = stored("/home/user/dl/t286/sweep_raw.jsonl")
rx = {**rx_d, **rx_l}; pr = {**pr_d, **pr_l}
out = {}
for name, model, panel in models:
    acc_tb, flops, parts = tb.est(model, n23=1, newton=1, nblind=0, use12=False, return_parts=True)
    tau = parts["t23"][0]
    m_tie, dm_tie = tb._margin(model, TRI, np.full(n, V1), np.full(n, V2), np.full(n, V2 - 1e-6),
                               np.zeros(n), -np.ones(n), -tb.dmu(np.full(n, V2 - 1e-6)))
    m, J, act = margin_and_grad_lows(model)
    z = m / np.maximum(J * SIG_LOW, 1e-12)
    flip = np.where(m > 0, ndtr(-z), 1 - ndtr(z))
    F = {"F1_tb": acc_tb, "F2_phi05": float(ndtr(-m / 0.5).mean()), "F3_phi2": float(ndtr(-m / 2).mean()),
         "F4_phi8": float(ndtr(-m / 8).mean()), "F5_neg_med": float((m < 0).mean()), "F6_neg_tie": float((m_tie < 0).mean()),
         "F7_logJm": float(np.log1p(J / np.maximum(np.abs(m), 1e-9)).mean()), "F8_lowflip": float(flip.mean()),
         "F9_rx": rx.get(name, np.nan), "F10_probe1": pr.get(name, np.nan)}
    out[name] = {"panel": panel, "truth": truth[name], "features": F, "tb_flops": flops,
                 "per_class": {"m_med_mean": float(m.mean()), "J_mean": float(J.mean()), "tau_mean": float(tau.mean()),
                               "act_mean": float(act.mean())}}
    print(name, panel, "truth %.4f" % truth[name], {k: round(v, 4) for k, v in F.items()}, flush=True)
json.dump({"sigma_low": SIG_LOW, "anchor": [V1, V2, V3M], "models": out}, open("results/features.json", "w"), indent=1)
