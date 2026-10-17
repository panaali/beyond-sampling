"""Accuracy and mechanism signatures of 2nd-argmax RNNs (ours vs zoo), all from numpy float64 weights.

Signatures (model-level, cheap, chosen before looking at our models):
  acc        Gaussian-input accuracy, n = 2^23 samples (fixed rng seed 12345, same inputs for every model of a T)
  t1         step-2 switch nearest x1 = x0: min_k |t_k - 1| over neurons holding x0, per chart sign(x0) = +/-,
             with t_k = -(Wh diag(g1) Wi)_k / Wi_k (understand_ladder README, step 2)
  n_store    number of neurons that hold |x0| after step 1 in each chart (sign(Wi_k) = sign(x0))
  bnd1       share of misclassified mass within 1 degree of a relevant true boundary (2nd-largest = largest or
             = 3rd-largest), bnd_med = median error angle in degrees
  dead       neurons that are never active on 2^16 inputs
usage: python evaluate.py              # evaluates all trained npz + zoo neighbours, writes results.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../data/ladder_training"))
N_ACC = 2**23
CHUNK = 2**20


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    act = np.zeros(Wh.shape[0], bool)
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
        act |= (h > 0).any(0)
    return h @ Wo.T, act


def signatures(model, n=N_ACC, seed=12345):
    Wi, Wh, Wo = model
    T = Wo.shape[0]
    rng = np.random.default_rng(seed)
    ok = 0; errs_ang = []; act = np.zeros(len(Wi), bool)
    for done in range(0, n, CHUNK):
        x = rng.standard_normal((min(CHUNK, n - done), T))
        lo, a = forward(model, x); act |= a
        srt = np.sort(x, 1)
        wrong = lo.argmax(1) != np.argsort(x, 1)[:, -2]
        ok += int((~wrong).sum())
        xs = srt[wrong]
        gap = np.minimum(xs[:, -1] - xs[:, -2], xs[:, -2] - xs[:, -3]) if T >= 3 else xs[:, -1] - xs[:, -2]
        errs_ang.append(np.degrees(np.arcsin(np.clip(gap / np.sqrt(2) / np.linalg.norm(xs, axis=1), 0, 1))))
    ang = np.concatenate(errs_ang)
    out = dict(acc=ok / n, bnd1=float((ang < 1.0).mean()) if len(ang) else 1.0,
               bnd_med=float(np.median(ang)) if len(ang) else 0.0, dead=int((~act).sum()))
    for sg, name in ((1, "pos"), (-1, "neg")):
        g1 = sg * Wi > 0
        t = -(Wh @ (g1 * Wi)) / np.where(np.abs(Wi) > 1e-9, Wi, np.nan)
        out[f"t1_{name}"] = float(np.nanmin(np.abs(t - 1)))
        out[f"n_store_{name}"] = int(g1.sum())
    out["n_store"] = "%d/%d" % tuple(sorted((out["n_store_pos"], out["n_store_neg"])))
    out["t1"] = max(out["t1_pos"], out["t1_neg"])  # worst chart: both charts need an x1~x0 switch
    return out


def zoo(H, T, seed, n_seqs=None):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(H, T, seed=seed, n_seqs=n_seqs)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0].copy(), sd["rnn.weight_hh_l0"].copy(), sd["linear.weight"].copy()


def ours(path):
    d = np.load(path)
    return (d["Wi"], d["Wh"], d["Wo"]), float(d["wall_seconds"])


def main():
    out_path = os.path.join(HERE, "results.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    zoo_sizes = set()
    for p in sorted(glob.glob(os.path.join(ROOT, "m_*.npz"))):
        H, T, tag, seed = os.path.basename(p)[2:-4].split("_")
        H, T, seed, log2n, scaled = int(H), int(T), int(seed), int(tag.rstrip("s")), tag.endswith("s")
        key = f"ours_{H}_{T}_{tag}_{seed}"
        ZH, ZT = (2, 3, 4, 5, 6, 8, 16, 32), (2, 3, 4, 5, 6, 8, 10)
        zoo_sizes |= {(h, t) for h in (H - 1, H, H + 1) for t in (T - 1, T, T + 1)
                      if h in ZH and t in ZT and (h == H or t == T)}
        if key in res:
            continue
        m, wall = ours(p)
        res[key] = dict(H=H, T=T, seed=seed, log2n=log2n, scaled_warmup=scaled, wall=wall, **signatures(m))
        print(key, res[key], flush=True)
        json.dump(res, open(out_path, "w"), indent=1)
    if "--no-zoo" in sys.argv:
        return
    for H, T in sorted(zoo_sizes):
        for seed in range(5):
            key = f"zoo_{H}_{T}_30_{seed}"
            if key in res:
                continue
            res[key] = dict(H=H, T=T, seed=seed, log2n=30, **signatures(zoo(H, T, seed)))
            print(key, res[key], flush=True)
            json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
