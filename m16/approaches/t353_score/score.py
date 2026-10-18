"""T355: second scorer of T353 (frozen estimator_h2) vs T317f on DEV3. Scoring only. usage (repo root):
python m16/approaches/t353_score/score.py SRC DEV3_WEIGHTS.npz DEV3_TRUTHS.json -> dev3.json, dev3_summary.json
SRC = claude/t292-oracle-floor @29c15c68 m16/approaches/budget_t317d/t351/ (hash prefixes from the freeze commit message)."""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SRC, WF, TF = sys.argv[1:4]
PFX = {"cont_f.py": "327dbb4a6ddc", "cont_g.py": "d34dde3699c6", "cont_h.py": "d27f607ab730", "cont_h2.py": "3e10011254ab",
       "estimator_h2.py": "fb64bdc13253"}
FULL = {}
for f, h in PFX.items():
    FULL[f] = hashlib.sha256(open(os.path.join(SRC, f), "rb").read()).hexdigest(); assert FULL[f].startswith(h), (f, FULL[f])
assert FULL["cont_f.py"] == "327dbb4a6ddcc2ad9ae4745ac0548ec121beee309461a2aaa1b67b47e955a1ab"   # = frozen T317f (T343)
sys.path.insert(0, SRC)
import numpy as np
import cont_f as F  # noqa: E402
import estimator_h2 as E  # noqa: E402
FPE = 5948.0
z = np.load(WF, allow_pickle=False); names = [str(n) for n in z["names"]]
fn = os.path.join(HERE, "dev3.json"); rows = json.load(open(fn)) if os.path.exists(fn) else {}
for i, n in enumerate(names):
    if n in rows:
        continue
    m = (z["Wi"][i], z["Wh"][i], z["Wo"][i]); ef, ff = F.est(m); eh, fh = E.est(m)
    rows[n] = dict(T317f=dict(est=float(ef), C=float(ff) / FPE), T353=dict(est=float(eh), C=float(fh) / FPE))
    print(n, json.dumps(rows[n]), flush=True); json.dump(rows, open(fn, "w"), indent=1)
T = json.load(open(TF)); T = T.get("truths", T); ns = names
p = np.array([T[n]["acc"] for n in ns]); se2 = np.mean([T[n].get("se", 0) ** 2 for n in ns]); pq = float(np.mean(p * (1 - p)))
out = dict(n=len(ns), sha256=FULL, arms={})
for a in ("T317f", "T353"):
    est = np.array([rows[n][a]["est"] for n in ns]); e = est - p; C = np.array([rows[n][a]["C"] for n in ns]); mse = float((e ** 2).mean() - se2)
    out["arms"][a] = dict(rms_pp=100 * np.sqrt(mse), bias_pp=100 * float(e.mean()), sd_pp=100 * float(e.std()), corr=float(np.corrcoef(est, p)[0, 1]),
                          C=float(C.mean()), C_min=float(C.min()), C_max=float(C.max()), F=mse * max(C.mean(), 1000) / pq, mse_x_C=1e4 * mse * C.mean())
out["T353_beats_T317f_MSExC"] = bool(out["arms"]["T353"]["mse_x_C"] < out["arms"]["T317f"]["mse_x_C"])
json.dump(out, open(os.path.join(HERE, "dev3_summary.json"), "w"), indent=1); print(json.dumps(out, indent=1))
