"""T280 models: target, zoo, and my own random draws (seeds per README). Activity = mean fraction of h_10 > 0."""
import numpy as np

_cache = {}


def _f(m):
    return (m.rnn.weight_ih_l0.detach().numpy()[:, 0].astype(np.float64),
            m.rnn.weight_hh_l0.detach().numpy().astype(np.float64), m.linear.weight.detach().numpy().astype(np.float64))


def target():
    if "t" not in _cache:
        from alg_zoo import example_2nd_argmax
        _cache["t"] = _f(example_2nd_argmax())
    return _cache["t"]


def frob(seed, mu=0.0):
    rng = np.random.default_rng(seed); out = []
    for i, W in enumerate(target()):
        R = rng.standard_normal(W.shape) + (mu if i == 1 else 0.0)
        out.append(R * np.linalg.norm(W) / np.linalg.norm(R))
    return tuple(out)


def row(seed):
    rng = np.random.default_rng(seed); Wi, Wh, Wo = target()
    wi = rng.choice([-1.0, 1.0], len(Wi)) * np.abs(Wi)
    R = rng.standard_normal(Wh.shape); wh = R * (np.linalg.norm(Wh, axis=1) / np.linalg.norm(R, axis=1))[:, None]
    R = rng.standard_normal(Wo.shape); wo = R * (np.linalg.norm(Wo, axis=1) / np.linalg.norm(R, axis=1))[:, None]
    return wi, wh, wo


def activity(model, n=200_000, seed=7):
    Wi, Wh, _ = model; x = np.random.default_rng(seed).standard_normal((n, 10)); h = np.zeros((n, len(Wi)))
    for t in range(10):
        h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
    return float((h > 0).mean())


def get(name):
    if name == "target":
        return target()
    if name.startswith("frob"):
        return frob(2800 + int(name[4:]))
    if name.startswith("row"):
        return row(2810 + int(name[3:]))
    if name.startswith("mu"):
        return frob(int(name[2:]), mu=0.25)          # mu2900..mu2999
    if name.startswith("zoo"):
        from alg_zoo import zoo_2nd_argmax
        H, T, sd = (int(v) for v in name[3:].split("_"))
        return _f(zoo_2nd_argmax(H, T, seed=sd))
    raise ValueError(name)
