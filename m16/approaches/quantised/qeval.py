"""T352: float64 accuracy of quantised models (qtrain.py checkpoints, or PTQ of M_16,10).
Each model: N fresh inputs on its own CUDA Philox stream (seed 3520000 + index), float64 forward with its activation
quantiser (inference form: relu / 1[u > 0] / s * clip(round(u / s), 0, L - 1)) and input grid; label = 2nd-largest
position of the (discretised) input, ties by the lower index.
usage: python qeval.py CKPT.pt [N] -> CKPT_eval.json   |   python qeval.py --ptq [N] -> ptq_eval.json"""
import json, math, os, sys
import numpy as np, torch as th
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DEV = os.environ.get("QDEV", "cuda"); LEVELS = {0: 0.0, 1: 0.0, 2: 15.0, 3: 255.0}


def q_inf(u, act, s):
    if act == 0:
        return th.relu(u)
    if act == 1:
        return (u > 0).to(u.dtype)
    return s * th.clamp(th.round(u / s), 0, LEVELS[act])


def disc(x, grid):
    if grid == 0:
        return x
    step = 6.0 / (grid - 1)
    return th.clamp(th.round((x + 3.0) / step), 0, grid - 1) * step - 3.0


def label(x):
    return (x - th.arange(10, device=x.device, dtype=x.dtype) * 1e-7).topk(2, 1).indices[:, 1]


def acc(Wi, Wh, Wo, act, s, grid, n, seed, B=1 << 22):
    g = th.Generator(device=DEV); g.manual_seed(seed); ok = 0; done = 0
    while done < n:
        m = min(B, n - done); x = disc(th.randn(m, 10, generator=g, device=DEV, dtype=th.float64), grid)
        h = th.zeros(m, 16, dtype=th.float64, device=DEV)
        for t in range(10):
            h = q_inf(h @ Wh.T + x[:, t:t + 1] * Wi[None], act, s)
        ok += int(((h @ Wo.T).argmax(1) == label(x)).sum()); done += m
    p = ok / n
    return dict(acc=p, se=math.sqrt(p * (1 - p) / n), n=n, correct=ok, seed=seed)


def calibrate(Wi, Wh, act, n=10 ** 6):
    """PTQ step: s = 99.99th percentile of the positive pre-activations (all steps, n inputs, float forward) / (L - 1)"""
    g = th.Generator(device=DEV); g.manual_seed(3529999); x = th.randn(n, 10, generator=g, device=DEV, dtype=th.float64)
    h = th.zeros(n, 16, dtype=th.float64, device=DEV); us = []
    for t in range(10):
        u = h @ Wh.T + x[:, t:t + 1] * Wi[None]; v = u[u > 0]; us.append(v[:: max(1, len(v) // 400000)]); h = th.relu(u)
    return float(th.quantile(th.cat(us), 0.9999)) / LEVELS[act]


if __name__ == "__main__":
    if sys.argv[1] == "--ptq":
        n = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 8
        sys.path.insert(0, os.path.join(HERE, "..", "..")); from arc_eval import Wi0, Wh0, Wo0  # noqa: E402
        W = [th.as_tensor(w, dtype=th.float64, device=DEV) for w in (Wi0, Wh0, Wo0)]; out = {}
        for k, (nm, a_) in enumerate((("orig_float", 0), ("orig_ptq_binary", 1), ("orig_ptq_int4", 2), ("orig_ptq_int8", 3))):
            s = calibrate(W[0], W[1], a_) if a_ >= 2 else 1.0
            out[nm] = dict(act=a_, s=s, **acc(*W, a_, s, 0, n, 3520900 + k)); print(nm, out[nm], flush=True)
        json.dump(out, open(os.path.join(HERE, "ptq_eval.json"), "w"), indent=1)
    else:
        f = sys.argv[1]; n = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 8
        d = th.load(f, weights_only=False); out = {}
        for k, sp in enumerate(d["meta"]["spec"]):
            W = [d[a][k].double().to(DEV) for a in ("Wi", "Wh", "Wo")]; s = float(d["s"][k])
            out[sp["name"]] = dict(sp, s=s, **acc(*W, sp["act"], s, sp["grid"], n, 3520000 + k)); print(sp["name"], out[sp["name"]], flush=True)
        json.dump(out, open(f.replace(".pt", "_eval.json"), "w"), indent=1)
