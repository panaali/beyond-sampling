"""T343 (pre-registered in README): independent check of T339 (spread-lows closure) on a fresh panel. Own scoring code; the
estimators are the frozen sources, hash-checked. usage (repo root):
python m16/approaches/t339_check/check.py SRC WEIGHTS.npz TRUTHS.json TAG  -> TAG.json (resumable), TAG_summary.json
SRC = folder holding cont_f_t317f.py and cont_g.py (claude/t286-cheap-frontier c0564f03 closure_floor/)."""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC, WF, TF, TAG = sys.argv[1:5]
SHA = {"cont_f_t317f.py": "327dbb4a6ddcc2ad9ae4745ac0548ec121beee309461a2aaa1b67b47e955a1ab",
       "cont_g.py": "d34dde3699c62d5784b2051d6ce8446ca03a0180e1ef6558ef117e3525b71605"}
for f, h in SHA.items():
    got = hashlib.sha256(open(os.path.join(SRC, f), "rb").read()).hexdigest()
    assert got == h, (f, got)
sys.path.insert(0, SRC)
import numpy as np
import cont_f_t317f as F  # noqa: E402
import cont_g as G  # noqa: E402
FPE = 5948.0
z = np.load(WF, allow_pickle=False); names = [str(n) for n in z["names"]]
TAB = {"0": G.TABLES, "+1": G._tables(1.0), "-1": G._tables(-1.0)}
fn = os.path.join(HERE, f"{TAG}.json"); rows = json.load(open(fn)) if os.path.exists(fn) else {}
for i, n in enumerate(names):
    if n in rows:
        continue
    m = (z["Wi"][i], z["Wh"][i], z["Wo"][i]); ef, ff = F.est(m); r = dict(T317f=dict(est=float(ef), C=float(ff) / FPE))
    for s, tab in TAB.items():
        e, f = G.est(m, tables=tab); r[s] = dict(est=float(e), C=float(f) / FPE)
        if s == "0":
            r["bitwise_s0"] = bool(e == ef and f == ff)
    rows[n] = r; print(n, json.dumps(r), flush=True); json.dump(rows, open(fn, "w"), indent=1)
if TF == "-":
    sys.exit(0)
T = json.load(open(TF)); T = T.get("truths", T)
ns = [n for n in names if n in rows]; p = np.array([T[n]["acc"] for n in ns]); se2 = np.array([T[n].get("se", 0.0) ** 2 for n in ns])
pq = float(np.mean(p * (1 - p))); out = dict(n=len(ns), bitwise_s0_all=all(rows[n]["bitwise_s0"] for n in ns), arms={})
for s in ("0", "+1", "-1"):
    est = np.array([rows[n][s]["est"] for n in ns]); e = est - p; C = np.array([rows[n][s]["C"] for n in ns]); mse = float((e ** 2).mean() - se2.mean())
    out["arms"][s] = dict(rms_pp=100 * np.sqrt(max(mse, 0)), rms_raw_pp=100 * float(np.sqrt((e ** 2).mean())), bias_pp=100 * float(e.mean()),
                          err_sd_pp=100 * float(e.std()), corr=float(np.corrcoef(est, p)[0, 1]), C=float(C.mean()), C_min=float(C.min()),
                          C_max=float(C.max()), F=mse * max(C.mean(), 1000) / pq, F_var_only=float(e.var()) * max(C.mean(), 1000) / pq)
e0 = np.array([rows[n]["0"]["est"] for n in ns]) - p
for s in ("+1", "-1"):
    sh = np.array([rows[n][s]["est"] - rows[n]["0"]["est"] for n in ns]); q = 1 - p
    A = np.c_[np.ones_like(q), q]; b = np.linalg.lstsq(A, sh, rcond=None)[0]; r2 = 1 - ((sh - A @ b) ** 2).sum() / ((sh - sh.mean()) ** 2).sum()
    A3 = np.c_[np.ones_like(q), q, e0]; b3 = np.linalg.lstsq(A3, sh, rcond=None)[0]; r23 = 1 - ((sh - A3 @ b3) ** 2).sum() / ((sh - sh.mean()) ** 2).sum()
    kern = {k: float(np.mean([sh[j] for j, n in enumerate(ns) if n.startswith(k)])) for k in sorted({n.split("_")[0] for n in ns})}
    out[f"shift_{s}"] = dict(mean_pp=100 * float(sh.mean()), sd_pp=100 * float(sh.std()), corr_with_1mp=float(np.corrcoef(sh, q)[0, 1]),
                            affine=[100 * float(b[0]), float(b[1])], affine_R2=float(r2), with_e0=[100 * float(b3[0]), float(b3[1]), float(b3[2])],
                            with_e0_R2=float(r23), corr_with_e0=float(np.corrcoef(sh, e0)[0, 1]), by_kernel_mean_pp={k: 100 * v for k, v in kern.items()})
json.dump(out, open(os.path.join(HERE, f"{TAG}_summary.json"), "w"), indent=1); print(json.dumps(out, indent=1))
