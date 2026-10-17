"""Configuration-local additive residuals on fixed central fibers."""
import numpy as np
from .enrich import cross, support_probes


class Additive:
    def __init__(self, oracle, shape, weights):
        self.oracle, self.shape, self.weights = oracle, tuple(shape), weights
        self.base = cross.TensorCross(oracle, list(shape), 2, 2)
        self.baseline_integral = self.base.integral(weights)
        self.work = dict(prediction_scalar_proxy=0, residual_subtractions=0,
                         contraction_proxy=0, probe_indices_constructed=0)
        points = support_probes(shape)
        self.work['probe_indices_constructed'] = len(points)*len(shape)
        residual = oracle(points)-self.base_predict(points)
        self.work['residual_subtractions'] += len(points)
        values = dict(zip(points, residual))
        self.terms = []
        for c in range(shape[0]):
            center = tuple([c]+[n//2 for n in shape[1:]])
            r0 = values[center]
            fibers = []
            for k in range(1, len(shape)):
                ps = []
                for i in range(shape[k]):
                    p = list(center); p[k] = i; ps.append(tuple(p))
                fibers.append(np.array([values[p] for p in ps]))
            self.terms.append((float(r0), fibers))
        # These are algorithm controls, charged in construction.
        defect = oracle(points)-self.predict(points)
        self.fiber_defect = float(np.max(abs(defect)))

    def base_predict(self, points):
        self.work['prediction_scalar_proxy'] += len(points)*sum(2*g.shape[0]*g.shape[2] for g in self.base.cores)
        return self.base.predict(points)

    def predict(self, points):
        ix = np.asarray(points, dtype=int)
        result = self.base_predict(ix)
        for c, (r0, fibers) in enumerate(self.terms):
            mask = ix[:, 0] == c
            if np.any(mask):
                result[mask] += sum(f[ix[mask, k+1]] for k, f in enumerate(fibers))-(len(fibers)-1)*r0
                self.work['prediction_scalar_proxy'] += int(mask.sum())*(len(fibers)+2)
        return result

    def integral(self):
        value = self.baseline_integral
        for c, (r0, fibers) in enumerate(self.terms):
            value += self.weights[0][c]*(sum(np.dot(w, f) for w, f in zip(self.weights[1:], fibers))-(len(fibers)-1)*r0)
            self.work['contraction_proxy'] += sum(2*len(f) for f in fibers)+len(fibers)+3
        return float(value)

    def report(self):
        return dict(unique_queries=len(self.oracle.cache), requests=self.oracle.requests,
                    original_cross_algebra_proxy=self.base.flops, enrichment_work=self.work.copy(),
                    configuration_terms=len(self.terms), central_fiber_defect=self.fiber_defect)
