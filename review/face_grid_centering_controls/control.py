"""T247 frozen analytic-grid controls. No network, random inputs or face walker."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from fractions import Fraction

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
START = time.process_time()
RESULT = {'complete': False, 'pass_all': False, 'rows': [], 'calls': {}}
ATTEMPT_OWNED = False


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save():
    OUT.mkdir(exist_ok=True)
    tmp = OUT / 'results.tmp'
    with tmp.open('w') as f:
        json.dump(RESULT, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUT / 'results.json')


def event(kind, **data):
    row = {'kind': kind, 'cpu': time.process_time() - START, **data}
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    assert time.process_time() - START <= 17, 'internal CPU cap'


def qrecord(q):
    return [str(q.numerator), str(q.denominator)]


def lattice(values):
    """Exact common dyadic denominator of stored binary64 scalars."""
    pairs = [float(v).as_integer_ratio() for v in values]
    exponent = max(d.bit_length() - 1 for _, d in pairs)
    return [n << (exponent - (d.bit_length() - 1)) for n, d in pairs], exponent


def law(g, c):
    """Exact-real inverse-linear law of stored arrays, including zero increments."""
    gi, ge = lattice(g)
    ci, ce = lattice(c)
    assert len(gi) == len(ci) >= 2
    assert all(b > a for a, b in zip(gi, gi[1:]))
    assert ci[0] == 0 and ci[-1] == 1 << ce
    diff = [b-a for a, b in zip(ci, ci[1:])]
    assert min(diff) >= 0
    # Flat CDF intervals have zero continuous latent mass; finite probes are separate.
    mean_n = sum(w*(a+b) for w, a, b in zip(diff, gi, gi[1:]))
    mean = Fraction(mean_n, 1 << (ce+ge+1))
    iz = gi.index(0)
    central = Fraction(ci[iz], 1 << ce)
    reflection = Fraction(max(abs(a+b-(1 << ce)) for a, b in zip(ci, reversed(ci))), 1 << ce)
    grid_reflection = Fraction(max(abs(a+b) for a, b in zip(gi, reversed(gi))), 1 << ge)
    return {'central': qrecord(central), 'central_gap': qrecord(central-Fraction(1, 2)),
            'mean': qrecord(mean), 'cdf_reflection_max': qrecord(reflection),
            'grid_reflection_max': qrecord(grid_reflection),
            'flat_intervals': diff.count(0), 'positive_intervals': sum(w>0 for w in diff),
            'intervals': len(diff), 'mass': qrecord(Fraction(sum(diff), 1 << ce))}


def unpack(row, key):
    return Fraction(*map(int, row[key]))


def trap(np, d, dg):
    weights = (d[:-1]+d[1:])/2.0
    cumulative = np.concatenate((np.array([0.0]), np.cumsum(weights)*dg))
    return cumulative/cumulative[-1], float(cumulative[-1]), weights


class Recorder:
    """Pass-through NumPy proxy; records the only source cumsum input/output."""
    def __init__(self, np):
        self.np, self.records = np, []
    def __getattr__(self, key):
        return getattr(self.np, key)
    def cumsum(self, a):
        value = self.np.cumsum(a)
        self.records.append((a.copy(), value.copy()))
        return value


def source_namespace(np, ndtr):
    tree = ast.parse((HERE/'frozen/core.py').read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_face_tables':
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('_g', '_dg') for t in node.targets):
            selected.append(node)
    assert len(selected) == 3
    RESULT['extracted_ast_sha256'] = hashlib.sha256(ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    proxy = Recorder(np)
    ns = {'np': proxy, 'ndtr': ndtr}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'frozen/core.py:selected', 'exec'), ns)
    return ns, proxy


def main():
    global ATTEMPT_OWNED
    assert not OUT.exists(), 'refuse overwrite'
    OUT.mkdir()
    ATTEMPT_OWNED = True
    event('start')
    p = json.loads((HERE/'protocol.json').read_text())
    gate = json.loads((HERE/'gate.json').read_text())
    for pins in (p['pins'], gate['pins']):
        for path, sha in pins.items():
            assert digest(path) == sha, path
    assert gate['execution_allowed'] is True and gate['source_review_commit'] != 'pending'
    review = json.loads(Path(gate['source_review_path']).read_text())
    assert digest(gate['source_review_path']) == gate['source_review_sha256']
    assert review['complete'] and review['pass_all'] and review['execution_allowed']
    required = [HERE/'control.py', HERE/'protocol.json', HERE/'frozen/core.py', HERE/'PROTOCOL.md']
    for path in required:
        assert review['pins'][str(path)] == digest(path)
    for path, sha in review['pins'].items():
        assert digest(path) == sha, path
    RESULT['protocol_sha256'] = digest(HERE/'protocol.json')
    RESULT['gate_sha256'] = digest(HERE/'gate.json')
    RESULT['limits'] = {k: list(resource.getrlimit(v)) for k, v in [('cpu', resource.RLIMIT_CPU), ('as', resource.RLIMIT_AS)]}
    save()
    assert all(0 < x <= 20 for x in RESULT['limits']['cpu'])
    assert all(0 < x <= 2*1024**3 for x in RESULT['limits']['as'])
    assert sys.executable == p['python'] and sys.prefix == p['prefix']
    assert all(os.environ.get(k) == '1' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'])
    import numpy as np
    import scipy
    from scipy.special import ndtr
    RESULT['runtime'] = {'python': sys.executable, 'prefix': sys.prefix, 'numpy': np.__version__, 'numpy_path': np.__file__, 'scipy': scipy.__version__, 'scipy_path': scipy.__file__}
    save()
    assert np.__version__ == p['numpy'] and scipy.__version__ == p['scipy']
    for name, density, expected_old, expected_trap in p['synthetics']:
        event('attempt', name=name)
        g = np.array([-1.0, 0.0, 1.0]); d = np.array(density, dtype=float)
        raw = np.cumsum(d)
        old = (raw-raw[0])/(raw[-1]-raw[0])
        fixed, z, weights = trap(np, d, 1.0)
        archive = OUT/(name+'.npz')
        np.savez(archive, g=g, d=d, original=old, trapezoid=fixed, trap_weights=weights)
        row = {'name': name, 'archive': archive.name, 'sha256': digest(archive), 'status': 'returned', 'trap_Z': z}
        RESULT['rows'].append(row); save(); event('returned', name=name)
        row['original'] = law(g, old); row['trapezoid'] = law(g, fixed)
        # Comparisons allow rounded rational CDF values for non-dyadic controls.
        row['passed'] = all(abs(unpack(row[k], 'central')-Fraction(*expected)) <= Fraction(1, 2**50) for k, expected in [('original', expected_old), ('trapezoid', expected_trap)])
        save(); event('checked', name=name, passed=row['passed']); assert row['passed']
    event('attempt', name='source_T2')
    ns, proxy = source_namespace(np, ndtr)
    tables, z23, z12 = ns['_face_tables'](2)
    assert len(proxy.records) == 1
    source_archive = OUT/'source_original.npz'
    np.savez(source_archive, g=ns['_g'], d=proxy.records[0][0], source_cumulative=proxy.records[0][1], original=tables['12'][1])
    RESULT['source_return'] = {'archive': source_archive.name, 'sha256': digest(source_archive), 'Z23': float(z23), 'Z12': float(z12)}
    save(); event('source_returned')
    assert len(proxy.records) == 1 and set(tables) == {'12'} and z23 == 0.0
    d, source_cumulative = proxy.records[0]
    g, dg = ns['_g'], float(ns['_dg'])
    original = tables['12'][1]
    fixed, trap_z, weights = trap(np, d, dg)
    probes = np.array([0.0, np.nextafter(0.0, 1.0), 0.25, np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0), 0.75, np.nextafter(1.0, 0.0), 1.0])
    orig_q = np.interp(probes, original, g)
    trap_q = np.interp(probes, fixed, g)
    archive = OUT/'source_T2.npz'
    np.savez(archive, g=g, d=d, source_cumulative=source_cumulative, original=original, trapezoid=fixed, trap_weights=weights, probes=probes, original_probe=orig_q, trapezoid_probe=trap_q)
    row = {'name': 'source_T2', 'status': 'returned', 'archive': archive.name, 'sha256': digest(archive), 'source_Z12': float(z12), 'source_Z23': float(z23), 'trap_Z': trap_z, 'dg_hex': dg.hex(), 'source_cumulative_calls': len(proxy.records), 'intervals': len(g)-1, 'probes': len(probes)}
    RESULT['rows'].append(row); save(); event('returned', name='source_T2')
    row['original'] = law(g, original); save()
    row['trapezoid'] = law(g, fixed); save()
    di, de = lattice(d)
    row['density_reflection_max'] = qrecord(Fraction(max(abs(a-b) for a,b in zip(di, reversed(di))), 1 << de))
    # erf(8) is only a rounded library diagnostic, never an interval certificate.
    row['analytic_integral_diagnostic'] = math.erf(8.0)
    row['source_Z_delta_diagnostic'] = float(z12)-row['analytic_integral_diagnostic']
    row['trap_Z_delta_diagnostic'] = trap_z-row['analytic_integral_diagnostic']
    row['passed'] = (len(g)==160001 and unpack(row['original'], 'central_gap') > Fraction(1,640000)
                     and unpack(row['original'], 'mean') < 0
                     and abs(unpack(row['trapezoid'], 'central_gap')) < Fraction(1,10**10)
                     and abs(unpack(row['trapezoid'], 'mean')) < Fraction(1,10**9))
    save(); event('checked', name='source_T2', passed=row['passed']); assert row['passed']
    RESULT['calls'] = {'source_face_tables_T2': 1, 'source_ndtr_arrays': 1, 'source_exp_arrays': 1, 'trapezoid_tables': 4, 'synthetic_original_cumsum': 3, 'np_interp_arrays': 2, 'np_interp_points': 18, 'exact_law_scans': 8, 'model': 0, 'face_points': 0, 'rng': 0}
    assert len(RESULT['rows']) == 4 and all(r['passed'] for r in RESULT['rows'])
    event('complete')
    RESULT['complete'] = RESULT['pass_all'] = True
    RESULT['cpu_main'] = time.process_time()-START
    save()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        RESULT['complete'] = RESULT['pass_all'] = False
        RESULT['fatal'] = {'type': type(exc).__name__, 'message': str(exc), 'cpu': time.process_time()-START}
        if ATTEMPT_OWNED:
            save()
            with (OUT/'progress.jsonl').open('a') as f:
                f.write(json.dumps({'kind': 'fatal', **RESULT['fatal']})+'\n'); f.flush(); os.fsync(f.fileno())
        raise
