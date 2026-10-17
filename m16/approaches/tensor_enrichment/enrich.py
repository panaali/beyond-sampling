"""Fixed residual-fiber enrichment; no access to validation tensors."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path('/workspace/clean-neurons')
SOURCE = ROOT / 'm16/approaches/tensor_cross/estimator.py'
spec = importlib.util.spec_from_file_location('_reviewed_cross', SOURCE)
cross = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross)


class CachedOracle:
    def __init__(self, evaluate):
        self.evaluate = evaluate
        self.cache = {}
        self.requests = 0

    def __call__(self, points):
        keys = [tuple(map(int, p)) for p in points]
        self.requests += len(keys)
        new = list(dict.fromkeys(p for p in keys if p not in self.cache))
        if new:
            self.cache.update(zip(new, map(float, self.evaluate(new))))
        return np.array([self.cache[p] for p in keys])


def support_probes(shape):
    center = [n // 2 for n in shape]
    points = set()
    for c in range(shape[0]):
        base = center.copy()
        base[0] = c
        points.add(tuple(base))
        for k in range(1, len(shape)):
            for i in range(shape[k]):
                p = base.copy()
                p[k] = i
                points.add(tuple(p))
    return sorted(points)


class Enriched:
    def __init__(self, oracle, shape, weights, weighted=True, rounds=2):
        self.oracle, self.shape, self.weights = oracle, tuple(shape), weights
        self.base = cross.TensorCross(oracle, list(shape), 2, 2)
        self.terms = []
        self.records = []
        self.work = dict(prediction_scalar_proxy=0, pivot_score_proxy=0,
                         fiber_normalization_divisions=0, contraction_proxy=0,
                         probe_indices_constructed=0)
        candidates = set(support_probes(shape))
        self.work['probe_indices_constructed'] = len(candidates) * len(shape)
        oracle(sorted(candidates))
        self.baseline_integral = self.base.integral(weights)
        for _ in range(rounds):
            points = sorted(candidates)
            residual = oracle(points) - self.predict(points)
            if weighted:
                importance = np.prod([weights[k][np.array(points)[:, k]]
                                      for k in range(len(shape))], axis=0)
            else:
                importance = np.ones(len(points))
            scores = importance * residual ** 2
            self.work['pivot_score_proxy'] += len(points) * (len(shape) + 3)
            selected = int(np.argmax(scores))
            pivot, amplitude = points[selected], float(residual[selected])
            if amplitude == 0:
                self.records.append(dict(stopped_zero=True, pivot=list(pivot)))
                break
            fibers = []
            fiber_points = []
            for k, n in enumerate(shape):
                ps = []
                for i in range(n):
                    p = list(pivot)
                    p[k] = i
                    ps.append(tuple(p))
                values = oracle(ps) - self.predict(ps)
                fibers.append(values / amplitude)
                self.work['fiber_normalization_divisions'] += n
                fiber_points.extend(ps)
            # At the pivot every normalized fiber is one; the new term
            # interpolates the OLD residual on each queried fiber.
            before = oracle(fiber_points) - self.predict(fiber_points)
            self.terms.append((amplitude, fibers))
            after = oracle(fiber_points) - self.predict(fiber_points)
            candidates.update(fiber_points)
            self.records.append(dict(pivot=list(pivot), amplitude=amplitude,
                score=float(scores[selected]), max_fiber_ratio=max(float(np.max(abs(f))) for f in fibers),
                fiber_residual_before=float(np.max(abs(before))),
                fiber_residual_after=float(np.max(abs(after))),
                integral=self.integral(), unique_queries=len(oracle.cache)))

    def predict(self, points):
        ix = np.asarray(points, dtype=int)
        result = self.base.predict(ix)
        self.work['prediction_scalar_proxy'] += len(ix) * sum(2 * g.shape[0] * g.shape[2] for g in self.base.cores)
        for amplitude, fibers in self.terms:
            result += amplitude * np.prod([f[ix[:, k]] for k, f in enumerate(fibers)], axis=0)
            self.work['prediction_scalar_proxy'] += len(ix) * (len(fibers) + 2)
        return result

    def integral(self):
        result = self.baseline_integral
        for amplitude, fibers in self.terms:
            result += amplitude * np.prod([np.dot(w, f) for w, f in zip(self.weights, fibers)])
            self.work['contraction_proxy'] += sum(2 * len(f) for f in fibers) + len(fibers)
        return float(result)

    def report(self):
        return dict(unique_queries=len(self.oracle.cache), requests=self.oracle.requests,
                    original_cross_algebra_proxy=self.base.flops, enrichment_work=self.work.copy(),
                    rounds=self.records)
