"""Zoo weights and helpers for T73 (understand_ladder): (Wi (H,), Wh (H,H), Wo (T,H)) numpy float64."""
import functools

import numpy as np


@functools.lru_cache(None)
def _load(H, T, seed):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(H, T, seed=seed)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0].copy(), sd["rnn.weight_hh_l0"].copy(), sd["linear.weight"].copy()


def zoo_model(H, T, seed):
    Wi, Wh, Wo = _load(H, T, seed)
    return Wi.copy(), Wh.copy(), Wo.copy()


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def mc_acc(model, n, rng, chunk=500_000):
    T = model[2].shape[0]
    ok = 0; done = 0
    while done < n:
        m = min(chunk, n - done)
        x = rng.standard_normal((m, T))
        ok += int((forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum()); done += m
    return ok / n


def ops_per_sample(H, T):
    """ARC convention (multiply-add = 1): T*H*(H+1) + T*H."""
    return T * H * (H + 1) + T * H


def noisy_abs(model, k, sigma, seed):
    """ARC protocol: absolute N(0, sigma^2) on every weight."""
    rng = np.random.default_rng(seed)
    return [tuple(w + sigma * rng.standard_normal(w.shape) for w in model) for _ in range(k)]


def noisy_rel(model, k, sigma, seed):
    """Leaderboard protocol: N(0, (sigma * RMS(W))^2) per matrix."""
    rng = np.random.default_rng(seed)
    return [tuple(w + sigma * np.sqrt(np.mean(w ** 2)) * rng.standard_normal(w.shape) for w in model) for _ in range(k)]


def best_seed(H, T):
    from alg_zoo.zoo import best_of_5_seed_2nd_argmax
    return best_of_5_seed_2nd_argmax(hidden_size=H, seq_len=T)


def leaderboard_models(H, T):
    """Same test models as m16/leaderboard.py models(H, T) (H, T) != (16, 10): zoo seeds 0-4 and N_NOISY = 6
    relative-sigma copies of the best-of-5 seed per sigma in SIGMAS, rng seed 0 (checked equal in run_t3.py)."""
    ms = {f"seed{s}": zoo_model(H, T, s) for s in range(5)}
    base = zoo_model(H, T, best_seed(H, T))
    for sigma in (0.0025, 0.005, 0.01, 0.02):
        for k, m in enumerate(noisy_rel(base, 6, sigma, 0)):
            ms[f"s{sigma}_{k}"] = m
    return ms
