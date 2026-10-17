"""Exact dyadic gate maps and integer projective double-description geometry.

Every topology decision uses Python integers/Fractions. Floating conversion is
outside this module. This represents real arithmetic with checkpoint weights.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from math import gcd, lcm
import time


class Cone:
    def __init__(self, rays, tight, constraints):
        self.rays = tuple(rays)
        self.tight = tuple(tight)
        self.constraints = constraints
        self.dim = len(self.rays[0])


class Exact:
    def __init__(self, seconds=110, max_bits=4096, max_rays=2000):
        self.deadline = time.process_time()+seconds
        self.max_bits = max_bits
        self.max_rays = max_rays
        self.counts = defaultdict(int)
        self.bits = defaultdict(int)

    def check(self):
        if time.process_time() > self.deadline:
            raise TimeoutError('exact geometry CPU deadline')

    def observe(self, values, kind, limit=True):
        largest = max((abs(int(x)).bit_length() for x in values), default=0)
        self.bits[kind] = max(self.bits[kind], largest)
        cap = self.max_bits if limit else 1000000
        if largest > cap:
            raise ArithmeticError(f'{kind} exceeded {cap} bits')

    def primitive(self, ray, kind='ray'):
        self.observe(ray, kind+'_before_gcd')
        g = 0
        for x in ray:
            g = gcd(g, abs(x))
            self.counts['gcd_calls'] += 1
        if not g:
            raise ArithmeticError('zero projective ray/normal')
        out = tuple(x//g for x in ray)
        self.observe(out, kind)
        return out

    def normal(self, fractions):
        denominator = 1
        for x in fractions:
            denominator = lcm(denominator, x.denominator)
            self.counts['lcm_calls'] += 1
            self.observe((x.numerator,), 'map_numerator')
            self.observe((x.denominator,), 'map_denominator')
        row = tuple(x.numerator*(denominator//x.denominator) for x in fractions)
        return self.primitive(row, 'normal') if any(row) else row

    def dot(self, a, b):
        self.counts['integer_dot_products'] += 1
        products = tuple(x*y for x, y in zip(a, b))
        self.observe(products, 'dot_products')
        result = sum(products)
        self.observe((result,), 'dot_sums')
        return result

    def det(self, matrix):
        self.check()
        self.counts['integer_determinants'] += 1
        a = [list(row) for row in matrix]
        n = len(a)
        if n == 0:
            return 1
        assert all(len(row) == n for row in a)
        previous, sign = 1, 1
        for k in range(n-1):
            pivot = next((i for i in range(k, n) if a[i][k]), None)
            if pivot is None:
                return 0
            if pivot != k:
                a[k], a[pivot] = a[pivot], a[k]
                sign = -sign
            for i in range(k+1, n):
                for j in range(k+1, n):
                    numerator = a[k][k]*a[i][j]-a[i][k]*a[k][j]
                    self.observe((numerator,), 'bareiss_numerator')
                    value, remainder = divmod(numerator, previous)
                    self.counts['bareiss_updates'] += 1
                    if remainder:
                        raise ArithmeticError('non-exact Bareiss division')
                    a[i][j] = value
                a[i][k] = 0
            previous = a[k][k]
        result = sign*a[-1][-1]
        self.observe((result,), 'determinant')
        return result

    def rank(self, matrix):
        self.check()
        self.counts['integer_rank_calls'] += 1
        a = [list(row) for row in matrix]
        if not a:
            return 0
        rank = 0
        for col in range(len(a[0])):
            pivot = next((i for i in range(rank, len(a)) if a[i][col]), None)
            if pivot is None:
                continue
            a[rank], a[pivot] = a[pivot], a[rank]
            for i in range(rank+1, len(a)):
                if not a[i][col]:
                    continue
                updated = [a[rank][col]*x-a[i][col]*y for x, y in zip(a[i], a[rank])]
                self.observe(updated, 'rank_intermediate')
                self.counts['rank_row_eliminations'] += 1
                if any(updated):
                    updated = self.primitive(updated, 'rank_row')
                a[i] = list(updated)
            rank += 1
            if rank == len(a):
                break
        return rank

    def clip(self, cone, normal, cid, want=(True, True)):
        self.check()
        self.counts['clips'] += 1
        if not any(normal):
            return (cone if want[0] else None), None
        assert cid not in cone.constraints
        values = [self.dot(normal, r) for r in cone.rays]
        positive = [i for i, x in enumerate(values) if x > 0]
        negative = [i for i, x in enumerate(values) if x < 0]
        zero = [i for i, x in enumerate(values) if x == 0]
        if not positive and not negative:
            raise ArithmeticError('nonzero plane annihilates full-dimensional cone')
        bit = 1 << cid
        new_rays, new_tight = [], []
        for p in positive:
            for m in negative:
                self.counts['adjacency_candidate_pairs'] += 1
                common = cone.tight[p] & cone.tight[m]
                if common.bit_count() < cone.dim-2:
                    continue
                self.counts['adjacency_containment_tests'] += len(cone.rays)
                if sum((z & common) == common for z in cone.tight) != 2:
                    continue
                ray = tuple(values[p]*x-values[m]*y for x, y in zip(cone.rays[m], cone.rays[p]))
                new_rays.append(self.primitive(ray))
                new_tight.append(common | bit)
                self.counts['intersections'] += 1
        def side(indices, orientation, enabled):
            if not indices or not enabled:
                return None
            rays = [cone.rays[i] for i in indices+zero]+new_rays
            tight = [cone.tight[i] for i in indices]+[cone.tight[i] | bit for i in zero]+new_tight
            if len(rays) > self.max_rays:
                raise ArithmeticError('ray cap')
            if len(set(rays)) != len(rays):
                raise ArithmeticError('duplicate extreme rays')
            constraints = dict(cone.constraints)
            constraints[cid] = tuple(orientation*x for x in normal)
            return Cone(rays, tight, constraints)
        return side(positive, 1, want[0]), side(negative, -1, want[1])

    def lift(self, cone, normal, cid):
        self.check()
        self.counts['lifts'] += 1
        a, alpha = normal[:-1], normal[-1]
        if not alpha:
            raise ValueError('lift needs nonzero input coefficient')
        sign = 1 if alpha > 0 else -1
        rays = [self.primitive(tuple(abs(alpha)*x for x in r)+(-sign*self.dot(a, r),)) for r in cone.rays]
        rays.append((0,)*cone.dim+(sign,))
        allmask = sum(1 << c for c in cone.constraints)
        tight = [z | (1 << cid) for z in cone.tight]+[allmask]
        constraints = {c: row+(0,) for c, row in cone.constraints.items()}
        constraints[cid] = normal
        return Cone(rays, tight, constraints)

    def matmul(self, left, right):
        self.check()
        columns = list(zip(*right))
        result = [[sum((x*y for x, y in zip(row, column)), Fraction(0)) for column in columns] for row in left]
        self.counts['rational_map_products'] += len(left)*len(columns)*len(right)
        for row in result:
            for x in row:
                self.observe((x.numerator,), 'map_numerator')
                self.observe((x.denominator,), 'map_denominator')
        return result

    def gates(self, model):
        wi, wh, wo = model
        H, T = len(wi), len(wo)
        if not any(wi):
            raise ValueError('all-zero Wi unsupported in initial lift reference')
        order = sorted(range(H), key=lambda k: -abs(wi[k]))
        roots = []
        for sign in (1, -1):
            A = [[wi[k] if sign*wi[k] > 0 else Fraction(0)] for k in range(H)]
            roots.append((Cone([(sign,)], [0], {0:(sign,)}), A))
        counts = [2]+[0]*(T-1)
        stack = [(cone, A, 1) for cone, A in roots[::-1]]
        final = []
        while stack:
            self.check()
            cone, A, t = stack.pop()
            if t == T:
                final.append((cone, A))
                continue
            B = self.matmul(wh, A)
            Z = [row+[wi[k]] for k, row in enumerate(B)]
            normals = [self.normal(row) for row in Z]
            first = order[0]
            cid = lambda k: 1+16*(t-1)+k
            pieces = []
            for sign in (1, -1):
                lifted = self.lift(cone, tuple(sign*x for x in normals[first]), cid(first))
                gates = [False]*H
                gates[first] = sign > 0
                pieces.append((lifted, gates))
            for neuron in order[1:]:
                new = []
                for current, gates in pieces:
                    pos, neg = self.clip(current, normals[neuron], cid(neuron))
                    if pos is not None:
                        active = gates.copy()
                        active[neuron] = True
                        new.append((pos, active))
                    if neg is not None:
                        new.append((neg, gates))
                pieces = new
            children = [(p, [Z[k] if g[k] else [Fraction(0)]*(t+1) for k in range(H)], t+1) for p, g in pieces]
            counts[t] += len(children)
            if sum(counts) > 20000:
                raise ArithmeticError('cell cap')
            stack.extend(children[::-1])
        self.counts['gate_cells'] = len(final)
        return final, counts

    def outcomes(self, cone, A, model, base_id=60):
        wo = model[2]
        T = len(wo)
        logits = self.matmul(wo, A)
        good = []
        for j in range(T):
            candidate = cone
            for other in range(T):
                if other == j:
                    continue
                margin = self.normal([x-y for x, y in zip(logits[j], logits[other])])
                if not any(margin):
                    if other < j:
                        self.counts['tie_candidates_rejected'] += 1
                        candidate = None
                        break
                    self.counts['zero_margins_skipped'] += 1
                    continue
                candidate, _ = self.clip(candidate, margin, base_id+other, (True, False))
                if candidate is None:
                    break
            if candidate is None:
                continue
            pieces = [(candidate, 0)]
            for other in range(T):
                if other == j:
                    continue
                normal = tuple(int(k == j)-int(k == other) for k in range(T))
                new = []
                for current, count in pieces:
                    below, above = self.clip(current, normal, base_id+T+other)
                    if below is not None:
                        new.append((below, count))
                    if above is not None and count == 0:
                        new.append((above, 1))
                pieces = new
            good.extend(p for p, count in pieces if count == 1)
        self.counts['correct_cones'] += len(good)
        return good

    def orthants(self, cone):
        pieces = [cone]
        for axis in range(cone.dim):
            normal = tuple(int(k == axis) for k in range(cone.dim))
            new = []
            for current in pieces:
                new.extend(side for side in self.clip(current, normal, 90+axis) if side is not None)
            pieces = new
        return pieces

    def triangulate(self, cone):
        out = []
        def recurse(face, dim):
            if len(face) == dim:
                if self.rank([cone.rays[i] for i in face]) != dim:
                    raise ArithmeticError('degenerate pulling simplex')
                return [face]
            vertex = face[0]
            simplices, seen = [], set()
            for cid in sorted(cone.constraints):
                bit = 1 << cid
                subset = tuple(i for i in face if cone.tight[i] & bit)
                if vertex in subset or not subset or subset in seen:
                    continue
                seen.add(subset)
                if len(subset) < dim-1 or self.rank([cone.rays[i] for i in subset]) != dim-1:
                    continue
                simplices.extend((vertex,)+simplex for simplex in recurse(subset, dim-1))
            return simplices
        for face in recurse(tuple(range(len(cone.rays))), cone.dim):
            rays = tuple(cone.rays[i] for i in face)
            determinant = self.det(rays)
            if not determinant:
                raise ArithmeticError('zero exact simplex determinant')
            denominators = tuple(sum(abs(x) for x in ray) for ray in rays)
            normalized_det = Fraction(abs(determinant), __import__('math').prod(denominators))
            out.append((rays, normalized_det))
        if not out:
            raise ArithmeticError('empty triangulation of full-dimensional cone')
        self.counts['simplices'] += len(out)
        return out

    def certify(self, cone):
        self.check()
        self.counts['certified_cones'] += 1
        if self.rank(cone.rays) != cone.dim:
            raise ArithmeticError('cone not full-dimensional')
        for ray, tight in zip(cone.rays, cone.tight):
            active = []
            actual = 0
            for cid, normal in cone.constraints.items():
                value = self.dot(normal, ray)
                if value < 0:
                    raise ArithmeticError('ray violates constraint')
                if value == 0:
                    active.append(normal)
                    actual |= 1 << cid
            if actual != tight:
                raise ArithmeticError('incomplete or false tight-set incidence')
            found = False
            for normals in combinations(active, cone.dim-1):
                cofactor = tuple((-1)**j*self.det([row[:j]+row[j+1:] for row in normals]) for j in range(cone.dim))
                if not any(cofactor):
                    continue
                projective = self.primitive(cofactor, 'cofactor_ray')
                if projective != ray and tuple(-x for x in projective) != ray:
                    raise ArithmeticError('cofactor ray certificate mismatch')
                found = True
                break
            if not found:
                raise ArithmeticError('no independent active-plane certificate')
            self.counts['cofactor_ray_certificates'] += 1


def dyadic_model(arrays):
    wi, wh, wo = arrays
    return ([Fraction.from_float(float(x)) for x in wi],
            [[Fraction.from_float(float(x)) for x in row] for row in wh],
            [[Fraction.from_float(float(x)) for x in row] for row in wo])
