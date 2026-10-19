"""T372 check (i): tiny RNN (T = 2 inputs, H = 3, C = 3), event {argmax = 0}, x ~ N(0, I_2).
P by the exact line walk in x1 (GL nodes in x2) vs a brute-force grid; dP/dW by boundary terms vs central FD of the
exact line-walk P (small delta) and of the brute-force grid P (large delta). -> results/tiny_check.json"""
import json
import os

import numpy as np
from scipy.special import ndtr

import lgrad
import rnn
import walk

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(372001)


def _draw():
    while True:                                   # first draw whose event probability is in (.2, .8), checked on a grid
        m = (rng.normal(size=3), rng.normal(size=(3, 3)) * .8, rng.normal(size=(3, 3)))
        g = np.linspace(-4, 4, 201); X = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        f = (rnn.forward(m, X).argmax(1) == 0).mean()
        if .2 < f < .8:
            return m


model = _draw()
NODES, WTS = np.polynomial.hermite_e.hermegauss(200); WTS = WTS / WTS.sum()     # x2 ~ N(0,1)
LO, HI = -9.0, 9.0


def lines():
    n = len(NODES); x0 = np.zeros((n, 2)); x0[:, 1] = NODES; d = np.zeros((n, 2)); d[:, 0] = 1.0
    return x0, d, np.zeros(n, int), np.full(n, LO), np.full(n, HI)


def P_line(m, grad=False):
    x0, d, k, lo, hi = lines(); r, a, b, _ = walk.argmax_pieces(m, x0, d, k, lo, hi); r, a, b = walk.merge(r, a, b)
    p = float(((ndtr(b) - ndtr(a)) * WTS[r]).sum())
    if not grad:
        return p
    g = np.zeros(len(rnn.flat(m)))
    for e, keep in ((a, a > LO + 1e-9), (b, b < HI - 1e-9)):
        G, _, _, _ = lgrad.endpoint_grads(m, x0, d, k, r[keep], e[keep])
        g += ((np.exp(-e[keep] ** 2 / 2) / np.sqrt(2 * np.pi) * WTS[r[keep]])[:, None] * G).sum(0)
    return p, g


def P_grid(m, n=3001):
    g = np.linspace(-7, 7, n); X = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    w = np.exp(-(X ** 2).sum(1) / 2) / (2 * np.pi) * (g[1] - g[0]) ** 2
    return float((w * (rnn.forward(m, X).argmax(1) == 0)).sum())


def main():
    sh = rnn.shapes(model); w0 = rnn.flat(model); p, g = P_line(model, True); out = dict(P_line=p, P_grid=P_grid(model))
    rows = []
    for i in range(6):
        u = rng.normal(size=len(w0)); u /= np.linalg.norm(u); h = 1e-5
        fd = (P_line(rnn.unflat(w0 + h * u, sh)) - P_line(rnn.unflat(w0 - h * u, sh))) / (2 * h)
        H = 0.05; fdg = (P_grid(rnn.unflat(w0 + H * u, sh)) - P_grid(rnn.unflat(w0 - H * u, sh))) / (2 * H)
        rows.append(dict(analytic=float(g @ u), fd_line=fd, fd_grid=fdg, rel_line=float(g @ u / fd - 1)))
        print(rows[-1], flush=True)
    out["dirs"] = rows; out["max_rel_line"] = max(abs(r["rel_line"]) for r in rows)
    print("P line %.6f grid %.6f  max rel (analytic vs FD line) %.2e" % (out["P_line"], out["P_grid"], out["max_rel_line"]))
    json.dump(out, open(os.path.join(HERE, "results", "tiny_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
