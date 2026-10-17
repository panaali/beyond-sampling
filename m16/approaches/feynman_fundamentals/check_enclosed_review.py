"""Independent high-precision GL20 check of T72's raw cubature enclosure."""
from pathlib import Path
from fractions import Fraction
from itertools import permutations
import hashlib
import importlib.util
import json
import time
import mpmath as mp

HERE = Path(__file__).resolve().parent
SOURCE = Path('/workspace/work/recover-small_h2_910/m16/approaches/small_ladder_review/rational_geometry')

def det4(rows):
    answer = 0
    for p in permutations(range(4)):
        inversions = sum(p[i] > p[j] for i in range(4) for j in range(i+1, 4))
        term = (-1)**inversions
        for i in range(4):
            term *= rows[i][p[i]]
        answer += term
    return answer

def main():
    started = time.process_time()
    mp.mp.dps = 70
    spec = importlib.util.spec_from_file_location('reviewed_enclosed_mass', SOURCE/'enclosed_mass.py')
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    model = author.EnclosedMass(20, seconds=19, verify_moments=False)
    nodes, weights = mp.gauss_quadrature(20, 'legendre')
    nodes = [(x+1)/2 for x in nodes]
    weights = [w/2 for w in weights]
    artifact = json.loads((SOURCE/'geometry_t4.json').read_text())
    cases = [('orthant', [[int(i == j) for j in range(4)] for i in range(4)], Fraction(1))]
    for index in (0, 128):
        record = artifact['correct_simplices'][index]
        rays = [list(map(int, ray)) for ray in record['integer_rays']]
        determinant = Fraction(*map(int, record['normalized_absolute_determinant']))
        cases.append((str(index), rays, determinant))
    rows = []
    for name, rays, determinant in cases:
        norms = [sum(abs(x) for x in ray) for ray in rays]
        check = Fraction(abs(det4(rays)), __import__('math').prod(norms))
        assert check == determinant
        endpoints = [[mp.mpf(x)/norm for x in ray] for ray, norm in zip(rays, norms)]
        raw, truncation = model.mass(rays, determinant)
        total = mp.mpf(0)
        for u, wu in zip(nodes, weights):
            for v, wv in zip(nodes, weights):
                for z, wz in zip(nodes, weights):
                    y = (u, (1-u)*v, (1-u)*(1-v)*z, (1-u)*(1-v)*(1-z))
                    q = sum(sum(y[j]*endpoints[j][axis] for j in range(4))**2 for axis in range(4))
                    total += wu*wv*wz*(1-u)**2*(1-v)/q**2
            if time.process_time()-started > 19:
                raise TimeoutError('independent review CPU budget')
        mass = total*(mp.mpf(determinant.numerator)/determinant.denominator)/(2*mp.pi**2)
        contained = mp.mpf(raw[0]) <= mass <= mp.mpf(raw[1])
        assert contained, (name, raw, mass)
        rows.append(dict(case=name, raw_interval=raw, high_precision_GL20=str(mass), contained=contained,
                         lower_margin=str(mass-raw[0]), upper_margin=str(raw[1]-mass), truncation_upper=truncation))
    result = dict(cpu_seconds=time.process_time()-started, precision_decimal_digits=70,
                  high_precision_quadrature_nodes=24000, independent_rule_generations=1,
                  model_passes=0, author_setup_counts=model.setup_counts, author_counts=dict(model.stats),
                  source_sha256=hashlib.sha256((SOURCE/'enclosed_mass.py').read_bytes()).hexdigest(),
                  artifact_sha256=hashlib.sha256((SOURCE/'geometry_t4.json').read_bytes()).hexdigest(), cases=rows,
                  scope='Numerical falsifier for raw cubature containment; not a standalone proof of all cones or IEEE assumptions.')
    (HERE/'enclosed_review_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
