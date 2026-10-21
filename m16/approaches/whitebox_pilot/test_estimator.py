"""Independent algebraic and numerical controls; not submission code."""
import math

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP

from estimator import Estimator, correct_means, first_mean_and_gains


def model(weights, seed=314):
    return MLP(width=weights[0].shape[0], depth=len(weights),
               weights=[fnp.asarray(w, dtype=fnp.float32) for w in weights], seed=seed)


def test_depth_one_exact_including_zero_neuron():
    weights = np.array([[1, -3, 0], [2, 0, 0], [-2, 4, 0]], dtype=np.float32)
    mlp = model([weights])
    with flops.BudgetContext(flop_budget=10**8, quiet=True) as ctx:
        got = Estimator(samples=17).predict(mlp, 10**8)
    expected = np.sqrt(np.sum(weights.astype(float)**2, axis=0) / (2 * math.pi))
    np.testing.assert_allclose(np.asarray(got)[0], expected, rtol=2e-7, atol=1e-8)
    assert ctx.flops_used > 0


def test_positive_later_weights_make_surrogate_exact():
    # Later ReLUs never switch: output is a positive linear map of h1.
    mlp = model([np.eye(3), np.ones((3, 3)), np.eye(3)])
    with flops.BudgetContext(flop_budget=10**8, quiet=True):
        means = [fnp.asarray(v, dtype=fnp.float64) for v in
                 ([0.2, 0.7, 0.9], [1.8, 1.8, 1.8], [1.8, 1.8, 1.8])]
        exact = fnp.ones(3, dtype=fnp.float32) / math.sqrt(2 * math.pi)
        gains = [fnp.ones(3, dtype=fnp.float32)] * 2
        got = correct_means(mlp, means, exact, gains)
    np.testing.assert_allclose(np.asarray(got)[-1], 3 / math.sqrt(2 * math.pi), rtol=2e-7)


def test_aggregated_correction_matches_explicit_shallow_network():
    rng = np.random.default_rng(70)
    weights = [rng.normal(size=(4, 4)).astype(np.float32) for _ in range(4)]
    mlp = model(weights)
    x = rng.normal(size=(53, 4)).astype(np.float32)
    activations = []
    for w in weights:
        x = np.maximum(x @ w, 0)
        activations.append(x)
    with flops.BudgetContext(flop_budget=10**9, quiet=True):
        mu, gains = first_mean_and_gains(mlp)
        means = [fnp.asarray(a.mean(axis=0, dtype=np.float64)) for a in activations]
        corrected = correct_means(mlp, means, mu, gains)
    # Independent dense surrogate construction with per-input evaluation.
    a = np.eye(4)
    centered = activations[0].astype(float) - np.asarray(mu)
    for layer in range(1, 4):
        a = (a @ weights[layer]) * np.asarray(gains[layer-1])
        explicit = (activations[layer] - centered @ a).mean(axis=0)
        np.testing.assert_allclose(np.asarray(corrected)[layer], explicit, rtol=2e-6, atol=1e-6)


def test_rng_repeatability_and_meter_charges_correction():
    rng = np.random.default_rng(9)
    mlp = model([rng.normal(size=(8, 8)).astype(np.float32) for _ in range(3)])
    out, costs = [], []
    for correct in (False, True, True):
        with flops.BudgetContext(flop_budget=10**8, quiet=True) as ctx:
            out.append(np.asarray(Estimator(samples=31, correction=correct).predict(mlp, 10**8)))
        costs.append(ctx.flops_used)
    np.testing.assert_array_equal(out[1], out[2])
    assert costs[1] == costs[2] > costs[0]
