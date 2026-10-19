"""Chunked-input RNN model class for T345's generic exact walk (ref/walks.py, unchanged; sha256 asserted).

usage: python cwalks.py     -> validation: k = 10 class vs walks.MLP and fast1 on T345's D1w128_s0; k = 1 vs walks.RNN
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{HERE}/ref")
for line in open(f"{HERE}/ref/SHA256SUMS"):
    h, f = line.split()
    assert hashlib.sha256(open(f"{HERE}/ref/{f}", "rb").read()).hexdigest() == h, f
import fast1  # noqa: E402
import walks  # noqa: E402

T = 10


def fpe(k, H):
    """Forward FLOPs (W_hh h_0 skipped since h_0 = 0)."""
    S = T // k
    return 2 * (T * H + (S - 1) * H * H + T * H) + S * H


class ChunkRNN:
    def __init__(self, k, Wih, Whh, Wout):
        self.k = k
        self.Wih, self.Whh, self.Wout = (np.asarray(a, dtype=np.float64) for a in (Wih, Whh, Wout))
        H, S = len(self.Whh), T // k
        self.H, self.S = H, S
        self.fwd_flops = fpe(k, H)
        # incremental cost model (as walks.RNN): a flip at step t is a rank-1 update into step t+1 (or into the
        # readout if t is the last step), then value+slope recompute of steps t+2..S-1 and of the readout
        step = 2 * (2 * H * H + 3 * H)
        self.suffix = np.repeat([(4 * H + max(S - 2 - t, 0) * step + 4 * T * H) if t < S - 1 else 4 * T
                                 for t in range(S)], H).astype(float)

    def pre_and_logits(self, x, d):
        h, dh, zs, dzs = np.zeros(self.H), np.zeros(self.H), [], []
        for t in range(self.S):
            sl = slice(t * self.k, (t + 1) * self.k)
            z, dz = self.Whh @ h + self.Wih @ x[sl], self.Whh @ dh + self.Wih @ d[sl]
            zs.append(z); dzs.append(dz)
            g = z > 0
            h, dh = np.where(g, z, 0.0), np.where(g, dz, 0.0)
        return np.concatenate(zs), np.concatenate(dzs), self.Wout @ h, self.Wout @ dh

    def forward(self, x):
        h = np.zeros((len(x), self.H))
        for t in range(self.S):
            h = np.maximum(h @ self.Whh.T + x[:, t * self.k:(t + 1) * self.k] @ self.Wih.T, 0)
        return h @ self.Wout.T

    def gates(self, x):
        """(N, S*H) bool gate pattern."""
        h, out = np.zeros((len(x), self.H)), []
        for t in range(self.S):
            z = h @ self.Whh.T + x[:, t * self.k:(t + 1) * self.k] @ self.Wih.T
            out.append(z > 0); h = np.maximum(z, 0)
        return np.concatenate(out, 1)


def load(tag):
    d = json.load(open(f"{HERE}/models/{tag}.json"))
    return ChunkRNN(d["k"], **d["weights"])


def validate():
    import subprocess
    w = json.loads(subprocess.check_output(["git", "show", "origin/claude/t345-ffn-control:"
                                             "m16/approaches/ffn_control/models/D1w128_s0.json"], cwd=HERE))["weights"]
    W1, W2 = np.asarray(w[0]), np.asarray(w[1])
    mine, ref = ChunkRNN(10, W1, np.zeros((128, 128)), W2), walks.MLP([W1, W2])
    assert mine.fwd_flops == ref.fwd_flops, (mine.fwd_flops, ref.fwd_flops)
    rng, md = np.random.default_rng(1), 0.0
    for _ in range(20):
        u, v = rng.standard_normal(T), rng.standard_normal(T)
        a = 0.5 * (walks.walk(mine, u, v, *walks.CAUCHY)[0] + walks.walk(mine, -u, -v, *walks.CAUCHY)[0])
        b = 0.5 * (walks.walk(ref, u, v, *walks.CAUCHY)[0] + walks.walk(ref, -u, -v, *walks.CAUCHY)[0])
        c = fast1.circle(W1, W2, u, v)[0]
        md = max(md, abs(a - b), abs(a - c))
    print("k=10 vs walks.MLP / fast1 on D1w128_s0: max |diff| over 20 circles", md)
    assert md < 1e-8
    rng = np.random.default_rng(2)
    H = 16
    Wi, Wh, Wo = rng.standard_normal(H), rng.standard_normal((H, H)) / 4, rng.standard_normal((T, H))
    mine, ref = ChunkRNN(1, Wi[:, None], Wh, Wo), walks.RNN(Wi, Wh, Wo)
    md = 0.0
    for _ in range(20):
        u, v = rng.standard_normal(T), rng.standard_normal(T)
        a, b = walks.walk(mine, u, v, *walks.CAUCHY), walks.walk(ref, u, v, *walks.CAUCHY)
        md = max(md, abs(a[0] - b[0])); assert a[2] == b[2]
    print("k=1 vs walks.RNN (random H=16 weights): max |diff| over 20 lines", md)
    assert md < 1e-10
    print("validation OK")


if __name__ == "__main__":
    validate()
