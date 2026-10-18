"""T234: accuracy and error anatomy of 2nd-argmax RNNs by plain MC on N(0, I_T) (float64 forward).

Error classes (m16/FINDINGS.md section 3): blind = global 2nd-largest <= 0 (any error there); otherwise an error is
'2-3' if the model picked the 3rd largest, '1-2' if it picked the largest, 'other' else.
usage: evaluate(models [(Wi, Wh, Wo) numpy], n, seed, device) -> list of dicts."""
import math

import numpy as np
import torch as th


@th.no_grad()
def evaluate(models, n=10 ** 8, seed=777, device="cuda", chunk=1 << 22):
    K = len(models); T = models[0][2].shape[0]
    W = [[th.as_tensor(w, dtype=th.float64, device=device) for w in m] for m in models]
    g = th.Generator(device=device); g.manual_seed(seed)
    cnt = np.zeros((K, 6), np.int64)      # correct, blind_total, blind_err, err23, err12, err_other
    done = 0
    while done < n:
        m = min(chunk, n - done)
        x = th.randn(m, T, generator=g, device=device, dtype=th.float64)
        order = x.argsort(-1)
        lab = order[:, -2]; top = order[:, -1]; third = order[:, -3]
        blind = x.gather(1, lab[:, None])[:, 0] <= 0
        for k, (Wi, Wh, Wo) in enumerate(W):
            h = th.zeros(m, Wi.shape[0], dtype=th.float64, device=device)
            for t in range(T):
                h = th.relu(h @ Wh.T + x[:, t:t + 1] * Wi[None])
            pred = (h @ Wo.T).argmax(1); err = pred != lab
            cnt[k] += [int((~err).sum()), int(blind.sum()), int((err & blind).sum()),
                       int((err & ~blind & (pred == third)).sum()), int((err & ~blind & (pred == top)).sum()),
                       int((err & ~blind & (pred != third) & (pred != top)).sum())]
        done += m
    out = []
    for k in range(K):
        c = cnt[k]; p = c[0] / n
        out.append(dict(n=n, seed=seed, acc=p, se=math.sqrt(p * (1 - p) / n), blind_share=c[1] / n,
                        err_blind=c[2] / n, err_23=c[3] / n, err_12=c[4] / n, err_other=c[5] / n,
                        blind_error_rate=c[2] / max(c[1], 1)))
    return out
