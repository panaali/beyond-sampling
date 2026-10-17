"""T59: reuse T55 terminal gate geometry; all model-specific work is charged.

This changes no polygon arithmetic or integration rule. Cached arrays are private
and read-only by convention; callers must not mutate the model after construction.
Not a certified probability or a complete challenge estimator.
"""
from collections import defaultdict
import itertools
import time
import numpy as np
from backward import FUND  # Installs the chosen read-only T55 helper directory.
from tail import clip
from moments import tail_mass_with_fallback


class BatchTerminal:
    def __init__(self, model):
        self.wi, self.wh, self.wo = (np.array(w, dtype=float, copy=True) for w in model)
        self.H = len(self.wi)
        self.T = len(self.wo)
        self.k = self.T - 2
        if self.T < 3 or self.wh.shape != (self.H, self.H) or self.wo.shape[1] != self.H:
            raise ValueError('invalid model dimensions')
        self.build = defaultdict(float)
        self.query = defaultdict(float)
        self.one = {}
        self.two = {}
        self.late = {}
        self.partitions = {}
        self.readouts = {}
        self.build_cpu = 0.
        self.query_cpu = 0.
        self.e0, self.e1, self.e2 = np.eye(3)

    def _validate(self, a, b, count, j):
        a = np.asarray(a, dtype=float)
        if a.shape != (self.H,) or np.any(a < 0) or not np.all(np.isfinite(a)):
            raise ValueError('need finite nonnegative hidden state')
        if not np.isfinite(b) or abs(b) > 1 + 1e-12 or int(j) != j or not 0 <= j < self.T:
            raise ValueError('invalid target coordinate/index')
        if count not in (0, 1, 2):
            raise ValueError('count must be 0,1,2')
        return tuple(a), float(b), int(j)

    def _step(self, regions, input_vector):
        next_regions = []
        for polygon, B in regions:
            Z = self.wh @ B + self.wi[:, None] * input_vector[None, :]
            self.build['affine_products'] += 1
            partial = [(polygon, np.zeros_like(B))]
            for neuron in range(self.H):
                if not np.any(Z[neuron]):
                    continue
                split = []
                for p, A in partial:
                    plus = clip(p, Z[neuron], self.build)
                    minus = clip(p, -Z[neuron], self.build)
                    if len(plus):
                        C = A.copy()
                        C[neuron] = Z[neuron]
                        split.append((plus, C))
                    if len(minus):
                        split.append((minus, A.copy()))
                partial = split
            next_regions.extend(partial)
        if len(next_regions) > 20000:
            raise ArithmeticError('tail region cap exceeded')
        self.build['constructed_step_cells'] += len(next_regions)
        return next_regions

    def _fresh(self, a, two):
        if a not in self.one:
            B = np.zeros((self.H, 3))
            B[:, 0] = a
            regions = [(np.diag([1., s1, s2]), B.copy())
                       for s1, s2 in itertools.product((-1, 1), repeat=2)]
            self.one[a] = self._step(regions, self.e1)
            self.build['one_fresh_complexes'] += 1
        if not two:
            return self.one[a]
        if a not in self.two:
            self.two[a] = self._step(self.one[a], self.e2)
            self.build['two_fresh_complexes'] += 1
        return self.two[a]

    def _prepare(self, a, b, j):
        key = ('early', a) if j <= self.T - 3 else ('late', a, b, j)
        readout_key = (key, j)
        if readout_key in self.readouts:
            self.build['readout_cache_hits'] += 1
            return self.readouts[readout_key]
        if j <= self.T - 3:
            regions = self._fresh(a, True)
        else:
            if key not in self.late:
                if j == self.T - 2:
                    regions = self._step(self._fresh(a, False), b * self.e0)
                    regions = self._step(regions, self.e2)
                else:
                    regions = self._step(self._fresh(a, True), b * self.e0)
                self.late[key] = regions
                self.build['late_complexes'] += 1
            regions = self.late[key]
        if key not in self.partitions:
            partition = 0.
            for polygon, _ in regions:
                partition += tail_mass_with_fallback(polygon, self.k, self.build)
                self.build['moment_calls'] += 1
                self.build['partition_moment_calls'] += 1
            if abs(partition - 1) > 1e-10:
                raise ArithmeticError(f'invalid partition mass {partition}')
            self.partitions[key] = partition
            self.build['max_partition_error'] = max(self.build['max_partition_error'], abs(partition - 1))
        correct = []
        for polygon, B in regions:
            q = polygon
            for other in range(self.T):
                if other == j:
                    continue
                margin = (self.wo[j] - self.wo[other]) @ B
                self.build['readout_margin_products'] += 1
                if not np.any(margin):
                    if other < j:
                        q = np.empty((0, 3))
                        break
                else:
                    q = clip(q, margin, self.build)
                if not len(q):
                    break
            if len(q):
                correct.append(q)
        self.readouts[readout_key] = correct
        self.build['readout_complexes'] += 1
        self.build['retained_readout_polygons'] += len(correct)
        return correct

    def probability(self, normalized_hidden, b, count, j):
        started = time.process_time()
        a, b, j = self._validate(normalized_hidden, b, count, j)
        self.query['queries'] += 1
        if count == 2:
            self.query['absorbed_count_two'] += 1
            self.query_cpu += time.process_time() - started
            return 0.
        # Validation/setup counts as query work; geometry including cache lookup
        # counts as build work. The two timings add to the full API CPU.
        before_build = time.process_time()
        self.query_cpu += before_build - started
        polygons = self._prepare(a, b, j)
        after_build = time.process_time()
        self.build_cpu += after_build - before_build
        below_x = b * self.e0 - self.e1
        below_y = b * self.e0 - self.e2
        conditions = [(below_x, below_y)] if count == 1 else [(-below_x, below_y), (below_x, -below_y)]
        result = 0.
        for q in polygons:
            for first, second in conditions:
                p = clip(clip(q, first, self.query), second, self.query)
                if len(p):
                    result += tail_mass_with_fallback(p, self.k, self.query)
                    self.query['moment_calls'] += 1
                    self.query['correct_pieces'] += 1
        if not -1e-12 <= result <= 1 + 1e-12:
            raise ArithmeticError(f'invalid terminal probability {result}')
        self.query_cpu += time.process_time() - after_build
        return float(result)

    def report(self):
        # Unique ndarray storage only, not Python containers or shared imports.
        arrays = {}
        for cache in (self.one, self.two, self.late):
            for regions in cache.values():
                for p, B in regions:
                    arrays[id(p)] = p
                    arrays[id(B)] = B
        for polygons in self.readouts.values():
            for p in polygons:
                arrays[id(p)] = p
        return dict(build_work=dict(self.build), query_work=dict(self.query),
                    build_cpu_seconds=self.build_cpu, query_cpu_seconds=self.query_cpu,
                    total_api_cpu_seconds=self.build_cpu + self.query_cpu,
                    cached_unique_arrays=len(arrays),
                    cached_array_bytes=sum(a.nbytes for a in arrays.values()),
                    partition_cache_entries=len(self.partitions),
                    readout_cache_entries=len(self.readouts))
