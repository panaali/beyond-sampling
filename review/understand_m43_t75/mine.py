"""T84 review helpers written for this review: weights straight from alg_zoo, ARC-protocol noise (absolute
N(0, sigma^2) on all 32 weights, draw order Wi, Wh, Wo per copy), own forward pass and own MC truth."""
import numpy as np


def zoo_w(seed):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(4, 3, seed=seed)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    assert set(sd) == {"rnn.weight_ih_l0", "rnn.weight_hh_l0", "linear.weight"}, sd.keys()   # no biases
    return sd["rnn.weight_ih_l0"][:, 0].copy(), sd["rnn.weight_hh_l0"].copy(), sd["linear.weight"].copy()


def noisy_w(base, k, sigma, seed):
    g = np.random.default_rng(seed); Wi, Wh, Wo = base; out = []
    for _ in range(k):
        a = Wi + sigma * g.standard_normal(4)
        b = Wh + sigma * g.standard_normal((4, 4))
        c = Wo + sigma * g.standard_normal((3, 4))
        out.append((a, b, c))
    return out


def fwd(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((x.shape[0], 4))
    for t in range(3):
        h = np.maximum(h @ Wh.T + x[:, [t]] * Wi, 0.0)
    return h @ Wo.T


def mc(model, n, seed, chunk=1_000_000):
    g = np.random.default_rng(seed); ok = 0; done = 0
    while done < n:
        m = min(chunk, n - done); x = g.standard_normal((m, 3))
        ok += int((fwd(model, x).argmax(1) == np.argsort(x, 1)[:, 1]).sum()); done += m
    return ok / n
