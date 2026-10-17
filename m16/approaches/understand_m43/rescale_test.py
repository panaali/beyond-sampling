"""Permanent invariance test (T85): a function-preserving positive rescaling of the hidden neurons (input row and
recurrent row x c_k, recurrent column and readout column / c_k, c_k = exp(U(-2, 2))) and a permutation of the hidden
neurons must not change a mechanism-derived estimate. usage: python rescale_test.py [n_models=20]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arc_m43_repro"))
import numpy as np
import est75 as E75
from m43 import Ops
from zoo43 import zoo_model, noisy_copies


def rescale(model, rng):
    Wi, Wh, Wo = model; H = len(Wi)
    c = np.exp(rng.uniform(-2, 2, H)); P = rng.permutation(H)
    Wi2 = (Wi * c)[P]; Wh2 = (Wh * c[:, None] / c[None, :])[np.ix_(P, P)]; Wo2 = (Wo / c[None, :])[:, P]
    return Wi2, Wh2, Wo2


def max_change(recipe, models, rng):
    worst = 0.0
    for m in models:
        a = E75.est75(m, Ops(), **recipe)
        for _ in range(3):
            worst = max(worst, abs(E75.est75(rescale(m, rng), Ops(), **recipe) - a))
    return worst


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = np.random.default_rng(12345)
    for name in ("RECIPE", "RECIPE85", "RECIPE85B", "RECIPE112"):
        rec = getattr(E75, name); out = []
        for s in range(5):
            ms = [zoo_model(s)] + noisy_copies(zoo_model(s), n, 0.05, 1)[:n]
            out.append(max_change(rec, ms, rng))
        print(f"{name:10s} max |estimate change| under rescaling+permutation, per zoo seed: " + ", ".join(f"{100*x:.2e}pp" for x in out), flush=True)
