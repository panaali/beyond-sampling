"""T294 stage 1 (pre-registered in README): mean-field ReLU closure on the 180 blind units of T293; compares with T293's sampled
contributions on DEV2 (and LB3). usage (repo root): python m16/approaches/meanfield/mf.py DEV2_WEIGHTS.npz -> mf.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(HERE, "..", "blind11"), os.path.join(ROOT, "m16")]
import numpy as np
from scipy.special import ndtr
import blind as B11                      # role moments, Clark closure, unit list conventions (T293, read-only reuse)
T = 10
PHI = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)


def propagate(model, mu_x, S_x, ops):
    """Joint Gaussian over v = (x (T), h (H)); returns mean/cov of the logits."""
    Wi, Wh, Wo = model; H = len(Wi); n = T + H
    mu = np.r_[mu_x, np.zeros(H)]; S = np.zeros((n, n)); S[:T, :T] = S_x
    for t in range(T):
        # z = Wh h + Wi x_t  as a linear map K (H, n) of v
        K = np.zeros((H, n)); K[:, T:] = Wh; K[:, t] += Wi
        mz = K @ mu; Czv = K @ S; Cz = Czv @ K.T; ops.n += 2 * H * n + 2 * H * n * n + 2 * H * H * n
        s = np.sqrt(np.maximum(np.diag(Cz), 1e-24)); a = mz / s; Pa = ndtr(a); pa = PHI(a)
        mh = mz * Pa + s * pa
        Eh2 = (mz ** 2 + s ** 2) * Pa + mz * s * pa; vh = np.maximum(Eh2 - mh ** 2, 0)
        Chv = Pa[:, None] * Czv                                                  # cov(h_new, v_old): Stein
        Chh = (Pa[:, None] * Pa[None, :]) * Cz; np.fill_diagonal(Chh, vh)        # first-order cross terms, exact diagonal
        # new state (x, h_new): x block unchanged; cov(h_new, x) = Chv[:, :T]
        S2 = np.zeros((n, n)); S2[:T, :T] = S[:T, :T]; S2[T:, :T] = Chv[:, :T]; S2[:T, T:] = Chv[:, :T].T; S2[T:, T:] = Chh
        mu = np.r_[mu[:T], mh]; S = S2; ops.n += 40 * H + 3 * H * H + 2 * H * T
    Ho = np.zeros((Wo.shape[0], n)); Ho[:, T:] = Wo
    mL = Ho @ mu; CL = Ho @ S @ Ho.T; ops.n += 2 * Wo.shape[0] * n * (n + Wo.shape[0])
    return mL, CL


def est_blind_mf(model):
    ops = B11.Ops(); ops.n = 0; acc = 0.0; units = []
    for a in range(T):
        for j in range(T):
            if a != j:
                units.append((None, a, j, 1 / 11 / 90))
    for q in range(T):
        for j in range(T):
            if j != q:
                units.append((q, None, j, 1 / 11 / 9))
    for case, a, j, w in units:
        mu_x, S_x = B11.moments(case, a, j)
        mL, CL = propagate(model, mu_x, S_x, ops)
        Bm = np.eye(len(mL))[j][None, :] - np.delete(np.eye(len(mL)), j, 0)
        acc += w * B11.clark_p_all_nonneg(Bm @ mL, Bm @ CL @ Bm.T, ops); ops.n += 2 * 9 * 10 * 10
    return 11 / 1024 * (1 - acc), float(ops.n)


if __name__ == "__main__":
    os.chdir(ROOT); import leaderboard as lb
    ref = json.load(open(os.path.join(HERE, "..", "blind11", "blind.json")))
    z = np.load(sys.argv[1], allow_pickle=False)
    panels = {"DEV2": [(str(n), (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))) for i, n in enumerate(z["names"])],
              "LB3": list(lb.models().items())}
    out = {}
    for pn, ms in panels.items():
        for name, m in ms:
            c, fl = est_blind_mf(m); s = ref[f"{pn}|{name}"]["sampled"]
            out[f"{pn}|{name}"] = dict(pred=c, sampled=s, cost_flops=fl)
            print(pn, name, "pred %.4f pp sampled %.4f pp diff %+.4f cost %.0f FPE" % (100 * c, 100 * s, 100 * (c - s), fl / 6000), flush=True)
        json.dump(out, open(os.path.join(HERE, "mf.json"), "w"), indent=1)
    for pn in panels:
        p = np.array([v["pred"] for k, v in out.items() if k.startswith(pn)]); s = np.array([v["sampled"] for k, v in out.items() if k.startswith(pn)])
        print(pn, "RMS %.3f pp mean %+.3f corr %.2f | pred sd %.3f sampled sd %.3f" % (100 * np.sqrt(((p - s) ** 2).mean()), 100 * (p - s).mean(),
              np.corrcoef(p, s)[0, 1], 100 * p.std(), 100 * s.std()))
