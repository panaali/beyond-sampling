"""T431 analysis on HP5 (group aggregates only). For each run: cost check (C >= B/2); primary F with the truth
variance subtracted (as T429): F = (sum e - sum SE^2) / sum p(1-p)/C; raw F = sum e / sum p(1-p)/C; the T281 ratio
sum e / sum (p(1-p)/C + SE^2); 95% CIs by bootstrap over models (one cluster per model; 10,000 resamples, seed 281);
bias, RMS, truth-variance share of sum e. usage: python t431_analyse.py -> results/hp5_t431.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ["PANEL"] = "hp5"
import score  # noqa: E402
FPE = score.FPE
names, man, models, tr = score.load_panel()
p = np.array([tr[n]["acc"] for n in names]); se2 = np.array([tr[n]["se"] ** 2 for n in names]); pq = p * (1 - p)
idx = np.random.default_rng(281).integers(0, len(p), (10_000, len(p)))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def summarise(est, cost, B, label, unit_fpe=False):
    """unit_fpe: the T429 recipes return cost in their own FPE (T(2H^2+2H) = 5,440 FLOPs per unit), used as charged
    (as T429; conservative vs FLOPs/5948). The control returns FLOPs (/5948)."""
    C = cost if unit_fpe else cost / FPE
    cost = C * FPE; e = (est - p) ** 2; s0 = pq / C
    Fsub = (e - se2)[idx].sum(1) / s0[idx].sum(1); Fraw = e[idx].sum(1) / s0[idx].sum(1)
    R281 = e[idx].sum(1) / (s0 + se2)[idx].sum(1)
    F = float((e - se2).sum() / s0.sum())
    return dict(label=label, budget=B, n=len(p), cost_ok=bool((cost >= 0.5 * B * FPE).all()),
                C_mean_fpe=float(C.mean()), C_min_fpe=float(C.min()),
                F=F, F_ci=ci(Fsub), F_raw=float(e.sum() / s0.sum()), F_raw_ci=ci(Fraw),
                R281=float(e.sum() / (s0 + se2).sum()), R281_ci=ci(R281),
                bias_pp=float(100 * (est - p).mean()), rms_pp=float(100 * np.sqrt(e.mean())),
                truth_share=float(se2.sum() / e.sum()),
                strict_ci=bool(ci(Fsub)[1] < 1), lenient_ci=bool(ci(Fsub)[1] < 10),
                strict_point=bool(F <= 1), lenient_point=bool(F <= 10))


def load(fn, B=None):
    rows = [json.loads(l) for l in open(os.path.join(score.D, fn))]
    if B is not None:
        rows = [r for r in rows if r["B"] == B]
    r = {x["k"]: x for x in rows}
    assert sorted(r) == list(range(len(p))), fn
    return np.array([r[k]["est"] for k in sorted(r)]), np.array([r[k]["cost"] for k in sorted(r)])


out = {}
for lab, fn, B in (("T10 4e6 (PRIMARY)", "raw_t431_t10_4000000.jsonl", 4_000_000),
                   ("T10 1e6", "raw_t431_t10_1000000.jsonl", 1_000_000),
                   ("T10 1e7", "raw_t431_t10_10000000.jsonl", 10_000_000),
                   ("isg 1e6", "raw_t431_isg_1000000.jsonl", 1_000_000)):
    f_ = os.path.join(score.D, fn)
    if os.path.exists(f_) and sum(1 for _ in open(f_)) == len(p):          # complete runs only
        out[lab] = summarise(*load(fn), B, lab, unit_fpe=True)
        e_, c_ = load(fn)
        out[lab + " [C in FLOPs/5948]"] = summarise(e_, c_ * 5440 / FPE, B, lab + " sensitivity", unit_fpe=True)
for B in (1_000_000, 4_000_000):
    out[f"control {B:.0e}"] = summarise(*load("raw_control.jsonl", B), B, f"plain MC control {B:.0e}")
json.dump(out, open(os.path.join(HERE, "results", "hp5_t431.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:22s} C {v['C_mean_fpe']:10.0f} ok={v['cost_ok']}  F {v['F']:.3f} [{v['F_ci'][0]:.3f}, {v['F_ci'][1]:.3f}]"
          f"  raw {v['F_raw']:.3f} [{v['F_raw_ci'][0]:.3f}, {v['F_raw_ci'][1]:.3f}]  R281 {v['R281']:.3f}"
          f"  bias {v['bias_pp']:+.4f} rms {v['rms_pp']:.4f} pp  truth share {v['truth_share']:.3f}")
