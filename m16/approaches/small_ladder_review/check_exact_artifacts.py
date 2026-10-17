"""Independent exact artifact checks; no author geometry or linear algebra code."""
from fractions import Fraction
from itertools import permutations
from math import prod, lcm
from pathlib import Path
import gzip
import hashlib
import json
import time

HERE = Path(__file__).resolve().parent
PERMS = [(p, (-1)**sum(p[i] > p[j] for i in range(4) for j in range(i+1, 4)))
         for p in permutations(range(4))]


def determinant(rays):
    return sum(sign*prod(rays[i][p[i]] for i in range(4)) for p, sign in PERMS)


def check(path):
    started = time.process_time()
    payload = path.read_bytes()
    data = json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)
    raw = data['model']
    wi = [Fraction(int(a), int(b)) for a, b in raw[0]]
    wh, wo = [[[Fraction(int(a), int(b)) for a, b in row] for row in matrix]
              for matrix in raw[1:]]
    import torch
    checkpoint = Path('/workspace/clean-neurons/work/checkpoints')/f'H{len(wi)}_T4_seed0.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == data['checkpoint_sha256']
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    assert wi == [Fraction(float(v)) for v in state['rnn.weight_ih_l0'].flatten()]
    for matrix, key in ((wh, 'rnn.weight_hh_l0'), (wo, 'linear.weight')):
        assert matrix == [[Fraction(float(v)) for v in row] for row in state[key]]
    denominator = lcm(*(x.denominator for x in wi + sum(wh, []) + sum(wo, [])))
    wi = [int(x*denominator) for x in wi]
    wh, wo = [[[int(x*denominator) for x in row] for row in matrix] for matrix in (wh, wo)]

    def forward(x):
        h, scale, signs = [0]*len(wi), 1, []
        for value in x:
            pre = [sum(w*v for w, v in zip(row, h))+w*value*scale for row, w in zip(wh, wi)]
            signs.extend((v > 0)-(v < 0) for v in pre)
            h = [max(v, 0) for v in pre]
            scale *= denominator
        return [sum(w*v for w, v in zip(row, h)) for row in wo], signs

    counts = {}
    for kind in ('gate_simplices', 'correct_simplices'):
        total = Fraction(0)
        for item in data[kind]:
            if time.process_time()-started > 19:
                raise TimeoutError('19CPU artifact budget')
            rays = [tuple(map(int, r)) for r in item['integer_rays']]
            det = Fraction(abs(determinant(rays)), prod(sum(abs(v) for v in r) for r in rays))
            assert det > 0
            assert det == Fraction(*map(int, item['normalized_absolute_determinant']))
            total += det
            # Positive combinations are interior to the full-dimensional simplex.
            for base in range(1, 8):
                center = [sum(base**i*r[i0] for i, r in enumerate(rays)) for i0 in range(4)]
                if len(set(center)) == 4:
                    break
            else:
                raise AssertionError('failed to find distinct input coordinates')
            logits, pattern = forward(center)
            winner = max(range(4), key=lambda j: (logits[j], -j))
            ordered = sorted(range(4), key=lambda j: (-center[j], j))
            above, target = ordered[:2]
            if kind == 'correct_simplices':
                assert winner == target
            for ray in rays:
                values, signs = forward(ray)
                assert all(s == 0 or s == c for s, c in zip(signs, pattern))
                if kind == 'correct_simplices':
                    assert all(values[target] >= value for value in values)
                    assert ray[above] >= ray[target]
                    assert all(ray[target] >= ray[j] for j in range(4) if j not in (above, target))
        if kind == 'gate_simplices':
            assert total == 16
        counts[kind] = len(data[kind])
    return dict(path=path.name, artifact_sha256=hashlib.sha256(payload).hexdigest(),
                checkpoint_sha256=data['checkpoint_sha256'], **counts,
                exact_gate_determinant_sum=16, all_determinants_match=True,
                exact_model_matches_checkpoint=True,
                all_ray_gate_patterns_match=True, all_correct_simplices_contained=True,
                cpu_seconds=time.process_time()-started,
                qualification='Containment and artifact consistency; not independent completeness/disjointness proof')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    name = 'geometry_t4.json' if args.hidden == 2 else 'geometry_h4_t4.json.gz'
    result = check(HERE/'rational_geometry'/name)
    (HERE/f'exact_artifact_h{args.hidden}_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
