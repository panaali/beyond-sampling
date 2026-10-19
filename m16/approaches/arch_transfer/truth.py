"""Plain-MC accuracy of a T432 model, float64 torch forward (equal to netnp to ~1e-13).

usage: python truth.py TAG SEED  -> truths/<TAG>_s<SEED>.json: acc_1e7 (inputs default_rng([432, 10]), 10 x 1e6) and
truth (1e8 inputs, default_rng([432, 11]), 100 x 1e6), each with its s.e.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch as th

th.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import archs  # noqa: E402
import load  # noqa: E402


def model(tag, seed):
    if tag in ("rnn32", "m16"):
        _, w = load.load(tag, seed)
        m = archs.RNN(w["Wh"].shape[0]).double()
        m.rnn.weight_ih_l0.data = th.tensor(w["Wi"]); m.rnn.weight_hh_l0.data = th.tensor(w["Wh"])
        m.linear.weight.data = th.tensor(w["Wo"])
        return m.eval()
    m = archs.build(tag).double()
    m.load_state_dict({k: v.double() for k, v in th.load(os.path.join(HERE, "weights", f"{tag}_s{seed}.pt")).items()})
    return m.eval()


def acc(m, key, nchunk, chunk=1_000_000, sub=50_000):
    rng = np.random.default_rng(key); ok = 0
    with th.no_grad():
        for _ in range(nchunk):
            X = rng.standard_normal((chunk, 10))
            for i in range(0, chunk, sub):
                x = th.tensor(X[i:i + sub])
                ok += (m(x).argmax(-1) == x.argsort(-1)[..., -2]).sum().item()
    p = ok / (nchunk * chunk)
    return p, float(np.sqrt(p * (1 - p) / (nchunk * chunk)))


if __name__ == "__main__":
    tag, seed = sys.argv[1], int(sys.argv[2])
    m = model(tag, seed)
    a7, s7 = acc(m, [432, 10], 10)
    a8, s8 = acc(m, [432, 11], 100)
    r = {"tag": tag, "seed": seed, "acc_1e7": a7, "se_1e7": s7, "truth": a8, "se": s8}
    os.makedirs(os.path.join(HERE, "truths"), exist_ok=True)
    json.dump(r, open(os.path.join(HERE, "truths", f"{tag}_s{seed}.json"), "w"))
    print(json.dumps(r))
