import numpy as np
def regions(m, T, n=200_000, seed=5):
    Wi, Wh, Wo = m; H = len(Wi); x = np.random.default_rng(seed).standard_normal((n, T)); h = np.zeros((n, H)); G = []
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0); G.append(h > 0)
    return len(np.unique(np.packbits(np.concatenate(G, 1), 1), axis=0))
