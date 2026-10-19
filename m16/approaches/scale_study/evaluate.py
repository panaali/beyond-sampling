"""Accuracy of a 2nd-argmax RNN by plain Monte Carlo on 1e7 N(0,1) inputs (T421).

The inputs are common to every model with the same T: np.random.default_rng([421, T]), 10 chunks of 1e6, so accuracies at
one grid point are paired. Forward pass in float64 numpy, h_0 = 0, h_t = relu(Wh h + Wi x_t), logits = Wo h_T.

usage: python evaluate.py weights/m_<H>_<T>_<stream><seed>.npz   -> prints JSON {acc, se, n}
"""
import json
import sys

import numpy as np

N, CHUNK = 10**7, 250_000  # 250k rows keeps H = 64 under the 2.5 GB cap; same input stream as 1e6-row chunks


def accuracy(Wi, Wh, Wo, n=N, chunk=CHUNK):
    T = Wo.shape[0]
    rng = np.random.default_rng([421, T])
    hits = 0
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T))
        h = np.zeros((chunk, len(Wi)))
        for t in range(T):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
        hits += int(((h @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    acc = hits / n
    return {"acc": acc, "se": (acc * (1 - acc) / n) ** 0.5, "n": n, "inputs": f"np.random.default_rng([421, {T}]), {n // chunk} x {chunk}"}


if __name__ == "__main__":
    d = np.load(sys.argv[1])
    print(json.dumps(accuracy(d["Wi"], d["Wh"], d["Wo"])))
