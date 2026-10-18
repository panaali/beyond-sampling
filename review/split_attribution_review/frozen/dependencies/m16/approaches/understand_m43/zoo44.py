"""Weights of alg_zoo.zoo_2nd_argmax(H, T, seed) and a generic forward / MC accuracy (diagnostic truth check)."""
import numpy as np


def zoo(H, T, seed):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(H, T, seed=seed)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0].copy(), sd["rnn.weight_hh_l0"].copy(), sd["linear.weight"].copy()


def mc_acc(model, n, rng, chunk=1_000_000):
    Wi, Wh, Wo = model; T = Wo.shape[0]; ok = 0; done = 0
    while done < n:
        m = min(chunk, n - done); x = rng.standard_normal((m, T)); h = np.zeros((m, len(Wi)))
        for t in range(T):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
        ok += int(((h @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]).sum()); done += m
    return ok / n


def ops_per_sample(H, T):
    return T * H * (H + 1) + T * H
