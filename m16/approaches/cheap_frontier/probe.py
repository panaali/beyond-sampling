"""Probe series m16/estimate.py (bisection probes at canonical inputs, model-free gap laws) with cheaper settings.
estimate.py is hard-wired to the original model (common.run, common.Wo); here its threshold functions are called
with the module globals pointed at the model under test. Knobs: version (1: 2~3 flips, 2: + 1~2 flips,
3: + first-order spread), bisection iters, position-balanced triple stride.
Labels: network answer on constructed inputs + bisection of the flip gap = deductive (exact computation of a
threshold of the network); flip probability from the model-free gap law = deductive approximation (canonical rest
values); blind error 8/9 = argued constant (no model call); triple subset (stride > 1) = INDUCTIVE.
Cost: 1 FPE per probe (one length-10 forward pass), as counted by estimate.py."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf import TRIPLES, cyclic_subset  # noqa: E402
import estimate as E  # noqa: E402  (m16/estimate.py)


def _bind(model):
    Wi, Wh, Wo = model

    def run(x):
        hs = np.zeros((len(x), x.shape[1] + 1, len(Wi)))
        for t in range(x.shape[1]):
            hs[:, t + 1] = np.maximum(hs[:, t] @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
        return hs
    E.run = run; E.Wo = Wo


def est_probe(model, version=2, iters=12, stride=1, seed=0, return_parts=False):
    _bind(model)
    sub = np.arange(720) if stride == 1 else cyclic_subset(stride)
    z = np.linspace(-3, 3, 13); w = np.exp(-z ** 2 / 2); w /= w.sum()
    evals = 0; parts = {}
    kinds = [("23", E.REL_GAP, E.threshold_general)] + ([("12", E.REL_GAP12, E.threshold_top)] if version >= 2 else [])
    for kind, gaps, fn in kinds:
        sg = np.sort(gaps); ps = []
        for j in sub:
            order = tuple(int(p) for p in TRIPLES[j])
            if version >= 3:
                g0, n = fn(order, seed, iters=iters); var = 0.0
                if 0 < g0 < 0.95:
                    for rk in range(3, 10):
                        eos = E.EOS.copy(); eos[rk] += 0.05 * E._SD[rk]
                        gp, n1 = fn(order, seed, iters=iters, eos=eos); n += n1
                        var += ((gp - g0) / 0.05) ** 2
                evals += n
                cdf = np.searchsorted(sg, np.clip(g0 + np.sqrt(var) * z, 0, None)) / len(sg)
                ps.append(float((w * cdf).sum()))
            else:
                th, n = fn(order, seed, iters=iters); evals += n
                ps.append(float(np.searchsorted(sg, th) / len(sg)))
        parts[kind] = float(np.mean(ps))
    parts["blind"] = (11 / 1024) * (8 / 9)
    err = parts["blind"] + (1 - 11 / 1024) * sum(parts[k] for k in ("23", "12") if k in parts)
    return (1 - err, float(evals), parts) if return_parts else (1 - err, float(evals))
