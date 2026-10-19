"""Own ground truth for T379: iid N(0,1) inputs, own float64 forward pass, chunked.
usage: python truth.py   (from this folder; 1e8 samples per clean model, 1e7 per noisy copy -> truths.json)"""
import json
import os
import sys
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

import models as M  # noqa: E402


def logits(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((x.shape[0], Wh.shape[0]))
    for t in range(x.shape[1]):
        h = h @ Wh.T + np.outer(x[:, t], Wi)
        np.maximum(h, 0, out=h)
    return h @ Wo.T


def n_correct(model, x):
    second = np.argpartition(x, -2, axis=1)[:, -2]       # index of the 2nd largest input
    return int((logits(model, x).argmax(1) == second).sum())


def truth(model, n, seed, chunk=1_000_000):
    rng = np.random.default_rng(seed); ok = 0; T = model[2].shape[0]
    for _ in range(n // chunk):
        ok += n_correct(model, rng.standard_normal((chunk, T)))
    p = ok / n
    return dict(acc=p, se=float(np.sqrt(p * (1 - p) / n)), n=n)


def check_torch(seed, n=1_000_000):
    """Own forward vs the alg_zoo torch module on the same inputs: argmax agreement."""
    import torch as th
    from alg_zoo.architectures import DistRNN
    sd = th.load(f"{M.ROOT}/data/t379_models/2nd_argmax_16_5_{2 ** 30}_{seed}.pth", weights_only=True)
    net = DistRNN(hidden_size=16, seq_len=5); net.load_state_dict(sd)
    x = np.random.default_rng(7).standard_normal((n, 5))
    with th.no_grad():
        a = net(th.tensor(x, dtype=th.float32)).argmax(-1).numpy()
    return float((a == logits(M.load(seed), x).argmax(1)).mean())


if __name__ == "__main__":
    out_path = "truths.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    names = sys.argv[1:] or None
    for name, m in M.all_models().items():
        if (names and name not in names) or name in out:
            continue
        clean = "/" not in name
        out[name] = truth(m, 100_000_000 if clean else 10_000_000, seed=zlib.crc32(name.encode()))
        if clean:
            out[name]["torch_agree"] = check_torch(int(name[1:]))
        print(name, out[name], flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
