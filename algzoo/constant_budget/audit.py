"""Independent scalar/feature audit; no source-model calls or fitting.

Run with the repository NumPy environment. The reference feature implementation
uses Python sorting and complex roots of unity, not the sweep's NumPy sorting
and trigonometric implementation. Only the sweep features function is extracted
with AST; importing sweep.py (and thus model-loading libraries) is unnecessary.
"""

import ast
import cmath
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np


ROOT = Path(__file__).resolve().parent


def independent_features(row):
    positions = sorted(range(9), key=lambda j: -row[j])
    p1, p2 = positions[:2]
    m1, m2 = row[p1], row[p2]
    out = [m1, m2, row[9]]
    for magnitude, position in ((m1 - m2, p1), (m2, p2)):
        for frequency in range(1, 5):
            root = cmath.exp(2j * math.pi * frequency * position / 9)
            out.extend((magnitude * root.real, magnitude * root.imag))
    out.append(m2 * cmath.exp(2j * math.pi * (p1 - p2) / 9).real)
    return out


def independent_predict(row, coefficients):
    phi = independent_features(row)
    return [max(0.0, math.fsum(phi[j] * coefficients[j][i]
                             for j in range(len(coefficients))))
            for i in range(len(coefficients[0]))]


def run():
    start = time.process_time()
    results = json.loads((ROOT / 'sweep_results.json').read_text())
    coef_path = ROOT / 'sweep_results_coefficients.json'
    coefficients = json.loads(coef_path.read_text())
    source = (ROOT / 'sweep.py').read_text()
    feature_node = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'features')
    module = ast.Module(body=[feature_node], type_ignores=[])
    scope = {'np': np}
    exec(compile(module, '<isolated sweep features>', 'exec'), scope)

    rng = random.Random(97131)
    sample = [[rng.gauss(0, 1) for _ in range(10)] for _ in range(128)]
    reference = np.array([independent_features(row) for row in sample])
    actual, names = scope['features'](np.array(sample))
    assert names == results['feature_names']
    assert len(names) == 20
    assert np.allclose(reference, actual, rtol=1e-12, atol=1e-12)
    feature_error = float(np.max(np.abs(reference - actual)))

    rows = []
    expected_keys = set()
    objective_evals = 0
    for model_key, model in results['models'].items():
        neurons = int(model_key.split('_')[0])
        assert [row['budget_per_neuron'] for row in model['results']] == results['budgets']
        threshold = 0.1 + 0.9 * (model['model_acc'] - 0.1)
        assert abs(model['mechanism_threshold'] - threshold) < 1e-14
        for row in model['results']:
            k = row['budget_per_neuron']
            key = f'{model_key}_k{k}'
            expected_keys.add(key)
            beta = coefficients[key]
            assert len(beta) == k and all(len(v) == neurons for v in beta)
            assert all(math.isfinite(c) for v in beta for c in v)
            assert row['total_fitted_scalars'] == k * neurons
            assert len(row['r2']) == neurons
            assert row['n_r2_at_least_0_8'] == sum(v >= 0.8 for v in row['r2'])
            expected_gap = (row['replacement_acc'] - 0.1) / (model['model_acc'] - 0.1)
            assert abs(row['gap_recovered'] - expected_gap) < 1e-14
            assert row['verdict_mechanism'] == (row['replacement_acc'] >= threshold)
            objective_evals += row['solver']['objective_evals'] + 1
            predictions = np.array([independent_predict(x, beta) for x in sample])
            vector_predictions = np.maximum(actual[:, :k] @ np.array(beta), 0)
            assert np.allclose(predictions, vector_predictions, rtol=1e-11, atol=1e-11)
            scale_errors = []
            for scale in (0.0, 0.1, 0.5, 2.0, 10.0):
                scaled = np.array([independent_predict([scale*v for v in x], beta)
                                   for x in sample])
                expected = scale * predictions
                assert np.allclose(scaled, expected, rtol=1e-11, atol=1e-11)
                scale_errors.append(float(np.max(np.abs(scaled - expected))))
            rows.append({'key': key, 'fitted_scalars': k * neurons,
                         'prediction_max_abs_error': float(np.max(np.abs(predictions-vector_predictions))),
                         'homogeneity_max_abs_error': max(scale_errors),
                         'solver_converged': row['solver']['success']})
    assert set(coefficients) == expected_keys
    expected_sequences = len(results['models']) * (results['n_train'] + results['n_test'])
    expected_validation = len(results['models']) * min(results['n_test'], 1024)
    assert results['compute']['model_hidden_forward_sequences'] == expected_sequences
    assert results['compute']['model_full_forward_sequences_validation'] == expected_validation
    assert results['compute']['regression_objective_evals'] == objective_evals
    assert results['train_seed'] != results['test_seed']
    return {
        'status': 'pass_for_checked_in_run',
        'coefficients_sha256': hashlib.sha256(coef_path.read_bytes()).hexdigest(),
        'sweep_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'synthetic_seed': 97131, 'synthetic_sequences': len(sample),
        'scales_checked': [0.0, 0.1, 0.5, 2.0, 10.0],
        'feature_max_abs_error': feature_error, 'rows': rows,
        'accounting_crosschecks': {
            'hidden_forward_sequences': expected_sequences,
            'validation_forward_sequences': expected_validation,
            'objective_evals_including_initial_mse_recheck': objective_evals,
        },
        'source_review': {
            'fit_split': 'Only train features/hidden targets enter fitting; independent seeded test data enter scoring.',
            'feature_dependence': 'Features use inputs and fixed structure only; no weights, fitted normalizers, or per-neuron feature selections.',
            'output_weights': 'Used only for replacement and model scoring, not regression fitting.',
            'fitted_parameter_count': 'Exactly K coefficients per neuron; no intercept. Fixed dictionary design is additional shared structure.',
        },
        'limitations': [
            'No model runs or independent accuracy reproduction; reported statistics were checked for internal consistency only.',
            'Seven of ten optimizers hit the iteration cap; failure of a fit does not prove failure of its hypothesis class.',
            'K=10 cuts the fourth gap-frequency cosine/sine pair, so budget effects also depend on feature ordering.',
            'Model-independent feature evaluation does not imply model-independent prior selection of this rank/position dictionary.',
            'Sequence counts and timing are not full FLOP accounting; least-squares work, feature operations, scoring, and solver algebra are not enumerated.',
            'Torch thread count is forced in source, but NumPy/BLAS thread limits depend on the launch environment.',
            'Changing K changes the available feature family as well as coefficient count; results do not establish a universal constant-budget limit.',
            'Homogeneity was checked at zero and positive scales, not negative scales, which need not preserve input ranks.',
        ],
        'audit_compute': {'model_calls': 0, 'fits': 0,
                          'process_cpu_seconds': time.process_time() - start},
    }


if __name__ == '__main__':
    audit = run()
    (ROOT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(audit['status'])
    print('Maximum feature error:', audit['feature_max_abs_error'])
    print('Maximum homogeneity error:', max(row['homogeneity_max_abs_error'] for row in audit['rows']))
