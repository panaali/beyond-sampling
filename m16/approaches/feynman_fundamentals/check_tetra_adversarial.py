"""Independent asymmetric-coalescence falsifier for T71's S3 boundary sum."""
from pathlib import Path
import importlib.util
import hashlib
import json
import time
import numpy as np
import mpmath as mp

HERE = Path(__file__).resolve().parent
SOURCE = Path('/workspace/work/recover-small_h2_56/m16/approaches/small_h2_56/odd_tetra/primitive.py')

def main():
    started = time.process_time()
    mp.mp.dps = 70
    spec = importlib.util.spec_from_file_location('reviewed_tetra', SOURCE)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    p = np.array([.8,.05,.05,.05,.05])
    vertices = np.array([p+10.**(-power)*(np.eye(5)[:4]-p) for power in range(2,9)])
    values, stats = author.inverse_fifth_tetra(vertices, known_plane_normal=np.ones(5))
    nodes, weights = mp.gauss_quadrature(12, 'legendre')
    nodes = [(x+1)/2 for x in nodes]
    weights = [w/2 for w in weights]
    rows = []
    for index, endpoints in enumerate(vertices):
        e = [[mp.mpf(float(x)) for x in row] for row in endpoints]
        reference = mp.mpf(0)
        for u, wu in zip(nodes, weights):
            for v, wv in zip(nodes, weights):
                for z, wz in zip(nodes, weights):
                    y = (u, (1-u)*v, (1-u)*(1-v)*z, (1-u)*(1-v)*(1-z))
                    q = sum(sum(y[j]*e[j][axis] for j in range(4))**2 for axis in range(5))
                    reference += wu*wv*wz*(1-u)**2*(1-v)/q**mp.mpf('2.5')
            if time.process_time()-started > 19:
                raise TimeoutError('tetra review CPU budget')
        _, single_stats = author.inverse_fifth_tetra(endpoints, known_plane_normal=np.ones(5))
        permuted, _ = author.inverse_fifth_tetra(endpoints[[3,1,0,2]], known_plane_normal=np.ones(5))
        scaled, _ = author.inverse_fifth_tetra(8*endpoints, known_plane_normal=np.ones(5)/8)
        rows.append(dict(epsilon=10.**(-index-2), value=float(values[index]), reference=str(reference),
                         relative_error=float(abs(mp.mpf(float(values[index]))/reference-1)),
                         analytic=single_stats['analytic_tetrahedra']==1,
                         cancellation_screen=single_stats['maximum_heuristic_relative_roundoff'],
                         singular_ratio=single_stats['minimum_singular_ratio'],
                         permutation_relative_difference=float(abs(permuted/values[index]-1)),
                         scale_relative_difference=float(abs(scaled*8**5/values[index]-1))))
    serial_stats = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in stats.items()}
    result = dict(cpu_seconds=time.process_time()-started, high_precision_tensor_nodes=7*12**3,
                  precision_decimal_digits=70, independent_rule_generations=1, model_passes=0,
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), cases=rows,
                  batch_stats=serial_stats, scope='Independent numerical adversarial check; not a directed rounding certificate.')
    (HERE/'tetra_adversarial_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
