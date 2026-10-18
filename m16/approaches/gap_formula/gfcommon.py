"""Shared helpers for T269: models, exact forward, summaries V of each input."""
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
T = 10
ANSWER_KEY_TARGET = 0.9531174789   # 3e11 samples, T192 gpu_cost_law/results/mc_target


def _load():
    t = np.load(os.path.join(_here, "../neuron_account/m16_weights.npz"))
    models = {"target": (t["Wi"], t["Wh"], t["Wo"])}
    z = np.load(os.path.join(_here, "zoo16_10_weights.npz"))
    for s in range(5):
        models[f"zoo{s}"] = (z[f"z{s}_Wi"], z[f"z{s}_Wh"], z[f"z{s}_Wo"])
    for sig in (0.005, 0.01):        # identical to m16/arc_eval.noisy_models(1, sig, seed=0)[0]
        rng = np.random.default_rng(0)
        models[f"noisy{sig}"] = tuple(W + sig * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape)
                                      for W in models["target"])
    return models


MODELS = _load()


def predict(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def summaries(x):
    """Dict of V components. Ages are 9 - position."""
    idx = np.argsort(-x, 1)
    s = np.take_along_axis(x, idx, 1)
    p = idx[:, :3]
    c = np.where(p[:, 0] == 9, 0, np.where(p[:, 1] == 9, 1, 2))     # x9 case: 0 max, 1 2nd, 2 below
    return {"M1": s[:, 0], "M2": s[:, 1], "M3": s[:, 2], "M4": s[:, 3], "p": p, "c": c,
            "K": (x > 0).sum(1), "target": idx[:, 1]}


def make(model, n, seed, chunk=500_000):
    """Fresh inputs: (err bool, summaries) concatenated over chunks."""
    rng = np.random.default_rng(seed)
    errs, Vs = [], []
    for i in range(0, n, chunk):
        x = rng.standard_normal((min(chunk, n - i), T))
        V = summaries(x)
        errs.append(predict(model, x) != V.pop("target"))
        V = {k: V[k] for k in ("M1", "M2", "M3", "c", "K")}
        V["c"], V["K"] = V["c"].astype(np.int8), V["K"].astype(np.int8)
        Vs.append(V)
    return np.concatenate(errs), {k: np.concatenate([v[k] for v in Vs]) for k in Vs[0]}
