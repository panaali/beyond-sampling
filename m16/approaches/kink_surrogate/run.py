"""T318 main run: 10 T297 DEV2 models; EXACT / SURR / CAP1 / CAP2 (three-line V1 and four-line), MC of closure c."""
import json
import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
import kink as K

Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
IDX = list(range(0, 80, 8))
T = 10


def predict(W, X):
    Wi, Wh, Wo = W
    h = np.zeros((len(X), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def mc_closure_c(W, n, seed):
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(n // 200_000):
        x = rng.standard_normal((200_000, T))
        o = np.argsort(-x, 1); v = np.take_along_axis(x, o, 1); r = np.arange(len(x))
        free = np.sort(o[:, 3:], 1); p4 = free[:, -1]
        mu4 = -norm.pdf(v[:, 3]) / ndtr(v[:, 3])
        xc = np.empty_like(x)
        xc[r[:, None], free] = mu4[:, None]
        xc[r, p4] = v[:, 3]
        for k in range(3):
            xc[r, o[:, k]] = v[:, k]
        ok += int((predict(W, xc) == o[:, 1]).sum())
    return ok / n


out = {}
for i in IDX:
    nm = str(Z["names"][i]); W = tuple(Z[k][i].astype(float) for k in ("Wi", "Wh", "Wo"))
    rec = {"mc_closure_c": mc_closure_c(W, 4_000_000, 318000 + i)}
    sets = {}
    for four in (False, True):
        for meth in ("EXACT", "SURR", "CAP1", "CAP2"):
            e, f, s = K.est(W, meth, four)
            tag = meth + ("_4L" if four else "")
            rec[tag] = {"est": e, "fpe": f / K.R.FPE_FLOPS}
            sets[tag] = s
    for four in ("", "_4L"):
        for meth in ("SURR", "CAP1", "CAP2"):
            rec[meth + four]["mismatch"] = {key: float(np.mean([K.mismatch(a, b) for a, b in zip(sets["EXACT" + four][key], sets[meth + four][key])]))
                                            for key in sets["EXACT" + four]}
    out[nm] = rec
    print(nm, "MC %.5f" % rec["mc_closure_c"], {t: round(100 * (v["est"] - rec["mc_closure_c"]), 3) for t, v in rec.items() if t != "mc_closure_c"}, flush=True)
json.dump(out, open("results/run.json", "w"), indent=1)
print("\nvariant: integration RMS / mean vs MC closure c (pp); mean FPE; RMS diff vs EXACT")
for tag in out[next(iter(out))]:
    if tag == "mc_closure_c":
        continue
    e = np.array([100 * (r[tag]["est"] - r["mc_closure_c"]) for r in out.values()])
    dx = np.array([100 * (r[tag]["est"] - r["EXACT" + ("_4L" if tag.endswith("_4L") else "")]["est"]) for r in out.values()])
    fpe = np.mean([r[tag]["fpe"] for r in out.values()])
    print(f"  {tag:10s} RMS {np.sqrt((e**2).mean()):.3f} mean {e.mean():+.3f} | FPE {fpe:.0f} | vs EXACT RMS {np.sqrt((dx**2).mean()):.3f}")
