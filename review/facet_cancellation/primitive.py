"""Exact oriented boundary-chain reduction. No model or geometry enumeration."""
from collections import Counter
from fractions import Fraction as F
from itertools import permutations
from math import gcd, prod


def pair(x):
    return [str(x.numerator), str(x.denominator)]


class Work:
    def __init__(self):
        self.counts = Counter()
        self.max_integer_bits = 0

    def det(self, rows):
        n = len(rows)
        self.counts['determinants'] += 1
        ans = 0
        for p in permutations(range(n)):
            z = 1
            for i in range(n):
                z *= rows[i][p[i]]
                self.counts['integer_multiplications'] += 1
            parity = sum(p[i] > p[j] for i in range(n) for j in range(i+1, n))
            ans += -z if parity % 2 else z
            self.counts['integer_accumulations'] += 1
            self.max_integer_bits = max(self.max_integer_bits, abs(z).bit_length(), abs(ans).bit_length())
        return ans

    def ray(self, raw, d):
        if not isinstance(raw, (list, tuple)) or len(raw) != d:
            raise ValueError('ray shape')
        if any(type(x) is not int for x in raw):
            raise ValueError('integer rays only')
        g = 0
        for x in raw:
            g = gcd(g, abs(x))
            self.counts['gcd_calls'] += 1
        if not g:
            raise ValueError('zero ray')
        self.counts['integer_ray_divisions'] += d
        return tuple(x//g for x in raw)


def reduce_chain(simplices, d, work, emit=lambda *args: None):
    if type(d) is not int or d not in (3, 4) or not isinstance(simplices, list) or not simplices:
        raise ValueError('nonempty dimension 3 or 4 chain required')
    facets = {}
    originals = []
    source_checks = []
    for index, raw in enumerate(simplices):
        emit('simplex_attempt', {'index': index})
        if not isinstance(raw, dict) or set(raw) != {'rays', 'expected_det'} or len(raw['rays']) != d:
            raise ValueError('simplex schema')
        rays = tuple(sorted(work.ray(r, d) for r in raw['rays']))
        if len(set(rays)) != d:
            raise ValueError('duplicate ray')
        orthant = []
        for j in range(d):
            signs = {1 if r[j] > 0 else -1 for r in rays if r[j]}
            if len(signs) != 1:
                raise ValueError('not a full dimensional single orthant')
            orthant.append(next(iter(signs)))
        orthant = tuple(orthant)
        det = work.det(rays)
        if det == 0:
            raise ValueError('degenerate simplex')
        norm = prod(sum(abs(x) for x in r) for r in rays)
        exact = F(abs(det), norm)
        if raw['expected_det'] is not None and exact != F(*map(int, raw['expected_det'])):
            raise ValueError('archived normalized determinant mismatch')
        orientation = 1 if det > 0 else -1
        expanded = F(0)
        ids = []
        for i in range(d):
            face = rays[:i] + rays[i+1:]
            key = (orthant, face)
            work.counts['facet_dictionary_lookups'] += 1
            if key not in facets:
                anchor_det = work.det((orthant,) + face)
                anchor_norm = d * prod(sum(abs(x) for x in r) for r in face)
                facets[key] = {'id': len(facets), 'orthant': orthant, 'rays': face,
                               'coefficient': 0, 'occurrences': 0,
                               'signed_det': F(anchor_det, anchor_norm)}
            rec = facets[key]
            coefficient = orientation * (-1 if i % 2 else 1)
            rec['coefficient'] += coefficient
            rec['occurrences'] += 1
            work.counts['facet_coefficient_adds'] += 1
            expanded += coefficient * rec['signed_det']
            work.counts['rational_identity_terms'] += 1
            ids.append([rec['id'], coefficient])
        record = {'index': index, 'det': pair(exact), 'expanded_det': pair(expanded), 'facets': ids}
        source_checks.append(record)
        emit('simplex_result', record)
        if expanded != exact:
            raise ArithmeticError('anchor determinant identity')
        originals.append({'rays': rays, 'weight': 1, 'det': pair(exact)})
    records = []
    terms = []
    for rec in facets.values():
        out = dict(rec, signed_det=pair(rec['signed_det']))
        records.append(out)
        if rec['coefficient'] and rec['signed_det']:
            terms.append({'facet_id': rec['id'], 'rays': (rec['orthant'],)+rec['rays'],
                          'weight': rec['coefficient'] * (1 if rec['signed_det'] > 0 else -1),
                          'det': pair(abs(rec['signed_det']))})
    return {'originals': originals, 'facets': records, 'terms': terms, 'source_checks': source_checks,
            'counts': dict(work.counts), 'observed_max_integer_bits': work.max_integer_bits}


def dyadic_totals(items, bits=128):
    """Per-term exact outward rounding prevents an enormous common denominator."""
    scale = 1 << bits
    lo = hi = absolute_upper = 0
    for row in items:
        value = F(*map(int, row['det'])) * row['weight']
        scaled = value * scale
        lo += scaled.numerator // scaled.denominator
        hi += -((-scaled.numerator) // scaled.denominator)
        a = abs(scaled)
        absolute_upper += -((-a.numerator) // a.denominator)
    return {'bits': bits, 'signed_lower_units': str(lo), 'signed_upper_units': str(hi),
            'absolute_upper_units': str(absolute_upper), 'term_count': len(items)}
