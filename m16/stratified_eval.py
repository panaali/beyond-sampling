"""Committed reproduction of the stratified-vs-sampling comparison (results/stratified_vs_sampling.json) plus the
analytic variance ratio that R1 showed is the reliable number. usage: python m16/stratified_eval.py"""
import json
import sys

import numpy as np

sys.path.insert(0, "m16")
import stratified as S  # noqa: E402
from arc_eval import Wi0, Wh0, Wo0, forward, noisy_models  # noqa: E402

d = json.load(open("m16/stratified_design.json")); P, alloc = np.array(d["P"]), np.array(d["alloc"])


def analytic_ratio(model, n_per=40_000, seed=0):
    """Exact-in-expectation variance ratio (plain / stratified) from per-stratum accuracies."""
    rng = np.random.default_rng(seed); acc = []
    for h in range(S.N_STRATA):
        x = S.draw(h, n_per, rng); acc.append((forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).mean())
    acc = np.array(acc); p = (P * acc).sum()
    var_strat = (P ** 2 * acc * (1 - acc) / alloc).sum()       # per unit of total budget
    return p * (1 - p) / var_strat


if __name__ == "__main__":
    for name, ms in [("orig", [(Wi0, Wh0, Wo0)]), ("s0.0025", noisy_models(6, 0.0025, 0)), ("s0.005", noisy_models(6, 0.005, 0))]:
        r = [analytic_ratio(m) for m in ms]
        print(f"{name}: variance ratio plain/stratified per model eval: mean {np.mean(r):.2f}, per copy {np.round(r, 2)}")
