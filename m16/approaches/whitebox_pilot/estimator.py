"""Exact-mean first-layer control variate; all array work is metered.

Only this file is an estimator entry point. The evaluation runner is separate.
"""
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


def first_mean_and_gains(mlp):
    w = mlp.weights[0]
    mu = fnp.sqrt(fnp.sum(w * w, axis=0) / (2.0 * fnp.pi))
    first_mean = mu
    variance = fnp.sum(w * w, axis=0) * (0.5 - 1.0 / (2.0 * fnp.pi))
    gains = []
    for w in mlp.weights[1:]:
        pre = mu @ w
        pre_var = fnp.maximum(variance @ (w * w), 1e-12)
        sigma = fnp.sqrt(pre_var)
        alpha = pre / sigma
        gain = flops.stats.norm.cdf(alpha).astype(fnp.float32)
        density = flops.stats.norm.pdf(alpha).astype(fnp.float32)
        mu = pre * gain + sigma * density
        second = (pre * pre + pre_var) * gain + pre * sigma * density
        variance = fnp.maximum(second - mu * mu, 0.0)
        gains.append(gain)
    return first_mean, gains


def sample_means(mlp, n, batch_size=2048):
    rng = fnp.random.default_rng(mlp.seed)
    sums = [fnp.zeros(mlp.width, dtype=fnp.float64) for _ in mlp.weights]
    for start in range(0, n, batch_size):
        x = rng.standard_normal((min(batch_size, n - start), mlp.width), dtype=fnp.float32)
        for layer, w in enumerate(mlp.weights):
            x = fnp.maximum(x @ w, 0.0)
            sums[layer] = sums[layer] + fnp.sum(x.astype(fnp.float64), axis=0)
    return [value / n for value in sums]


def correct_means(mlp, means, first_mean, gains):
    # Use float64 for small differences between sample and analytic means.
    delta = means[0] - first_mean
    rows = [first_mean]
    for i, (w, gain) in enumerate(zip(mlp.weights[1:], gains), start=1):
        delta = (delta @ w) * gain
        rows.append(means[i] - delta)
    return fnp.stack(rows).astype(fnp.float32)


class Estimator(BaseEstimator):
    def __init__(self, samples=None, correction=True):
        self.samples = samples
        self.correction = correction

    def predict(self, mlp, budget):
        # Dimensions/budget are metadata, not numerical model data. Reserve a
        # conservative setup allowance when selecting the default sample count.
        w, d = mlp.width, mlp.depth
        per_sample = 16 * w + d * (2 * w * w + 8 * w)
        setup_allowance = d * (12 * w * w + 512 * w)
        n = self.samples
        if n is None:
            n = max(1, int((0.1 * budget - setup_allowance) / per_sample))
        if n < 1:
            raise ValueError("samples must be positive")
        if self.correction:
            first_mean, gains = first_mean_and_gains(mlp)
        means = sample_means(mlp, n)
        if not self.correction:
            return fnp.stack(means).astype(fnp.float32)
        return correct_means(mlp, means, first_mean, gains)
