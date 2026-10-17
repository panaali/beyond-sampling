"""Fixed, nested fitted-scalar sweep; see README for prespecified design."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from alg_zoo import example_2nd_argmax, zoo_2nd_argmax

ROOT = Path(__file__).resolve().parent
BUDGETS = (1, 3, 5, 10, 20)


def features(x):
    p = x[:, :9].astype(np.float64)
    order = np.argsort(-p, axis=1)
    rows = np.arange(len(x))
    p1, p2 = order[:, 0], order[:, 1]
    m1, m2 = p[rows, p1], p[rows, p2]
    cols, names = [m1, m2, x[:, 9]], ['m1', 'm2', 'x9']
    for name, value, pos in [('gap', m1 - m2, p1), ('m2', m2, p2)]:
        for freq in range(1, 5):
            theta = 2 * np.pi * freq * pos / 9
            for trig in (np.cos, np.sin):
                cols.append(value * trig(theta))
                names.append(f'{name}*{trig.__name__}(2*pi*{freq}*p{1 if name == "gap" else 2}/9)')
    cols.append(m2 * np.cos(2 * np.pi * (p1 - p2) / 9))
    names.append('m2*cos(2*pi*(p1-p2)/9)')
    return np.column_stack(cols), names


def targets(model, x):
    out = []
    with torch.inference_mode():
        for start in range(0, len(x), 8192):
            _, h = model.rnn(torch.from_numpy(x[start:start+8192]).unsqueeze(-1))
            out.append(h[0].numpy())
    return np.concatenate(out).astype(np.float64)


def wilson(success, n):
    p, z = success / n, 1.959963984540054
    den = 1 + z*z/n
    middle = (p + z*z/(2*n)) / den
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [float(middle-radius), float(middle+radius)]


def fit(phi, h):
    initial = np.linalg.lstsq(phi, h, rcond=None)[0]
    def objective(flat):
        beta = flat.reshape(initial.shape)
        pre = phi @ beta
        error = np.maximum(pre, 0) - h
        loss = np.mean(error**2)
        grad = 2 * phi.T @ (error * (pre > 0)) / error.size
        return loss, grad.ravel()
    result = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                      options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-7})
    beta = result.x.reshape(initial.shape)
    return beta, {'iterations': int(result.nit), 'objective_evals': int(result.nfev),
                  'success': bool(result.success), 'message': str(result.message),
                  'initial_mse': float(objective(initial.ravel())[0]),
                  'train_mse': float(result.fun)}


def run(n_train, n_test):
    torch.set_num_threads(1)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    train = np.random.default_rng(71101).standard_normal((n_train, 10)).astype(np.float32)
    test = np.random.default_rng(71102).standard_normal((n_test, 10)).astype(np.float32)
    phi_train, names = features(train)
    phi_test, _ = features(test)
    truth = np.argsort(test, axis=1)[:, -2]
    result = {'train_seed': 71101, 'test_seed': 71102, 'n_train': n_train, 'n_test': n_test,
              'feature_names': names, 'budgets': list(BUDGETS), 'models': {},
              'torch_version': torch.__version__, 'numpy_version': np.__version__}
    coefficients = {}
    for key, loader in [('16_10', example_2nd_argmax), ('32_10', lambda: zoo_2nd_argmax(32, 10))]:
        model = loader().eval()
        h_train, h_test = targets(model, train), targets(model, test)
        output = model.linear.weight.detach().numpy().astype(np.float64)
        model_correct = (h_test @ output.T).argmax(1) == truth
        model_acc = float(model_correct.mean())
        # Check the split recurrence/output scoring path against the model API.
        with torch.inference_mode():
            direct = model(torch.from_numpy(test[:1024])).numpy()
        assert np.array_equal(direct.argmax(1), (h_test[:1024] @ output.T).argmax(1))
        digest = hashlib.sha256()
        for tensor in model.state_dict().values():
            digest.update(tensor.detach().numpy().tobytes())
        rows, last_correct = [], None
        variance = ((h_test-h_test.mean(0))**2).sum(0)
        for budget in BUDGETS:
            c0 = time.process_time()
            beta, solver = fit(phi_train[:, :budget], h_train)
            prediction = np.maximum(phi_test[:, :budget] @ beta, 0)
            error = ((h_test-prediction)**2).sum(0)
            r2 = np.where(variance > 1e-9, 1-error/np.maximum(variance, 1e-9), (error < 1e-6).astype(float))
            correct = (prediction @ output.T).argmax(1) == truth
            acc = float(correct.mean())
            delta = None
            if last_correct is not None:
                paired = correct.astype(float)-last_correct.astype(float)
                se = paired.std(ddof=1)/np.sqrt(n_test)
                delta = {'accuracy_change': float(paired.mean()),
                         'paired_95ci': [float(paired.mean()-1.96*se), float(paired.mean()+1.96*se)]}
            row = {'budget_per_neuron': budget, 'total_fitted_scalars': int(beta.size),
                   'replacement_acc': acc, 'replacement_acc_95ci': wilson(correct.sum(), n_test),
                   'gap_recovered': (acc-0.1)/(model_acc-0.1), 'r2': r2.tolist(),
                   'n_r2_at_least_0_8': int((r2 >= 0.8).sum()),
                   'mean_r2': float(r2.mean()), 'median_r2': float(np.median(r2)),
                   'verdict_mechanism': acc >= 0.1+0.9*(model_acc-0.1),
                   'change_from_previous_budget': delta, 'solver': solver,
                   'fit_and_score_cpu_seconds': time.process_time()-c0}
            rows.append(row)
            coefficients[f'{key}_k{budget}'] = beta.tolist()
            last_correct = correct
            print(key, budget, f'acc={acc:.5f}', f'R2>=.8={row["n_r2_at_least_0_8"]}',
                  f'fit={solver["message"]}', flush=True)
        result['models'][key] = {'model_acc': model_acc, 'model_acc_95ci': wilson(model_correct.sum(), n_test),
                                 'mechanism_threshold': 0.1+0.9*(model_acc-0.1),
                                 'weights_sha256': digest.hexdigest(), 'results': rows}
    result['compute'] = {'model_hidden_forward_sequences': 2*(n_train+n_test),
                         'model_full_forward_sequences_validation': 2*min(n_test, 1024),
                         'regression_objective_evals': sum(row['solver']['objective_evals']+1 for m in result['models'].values() for row in m['results']),
                         'process_cpu_seconds': time.process_time()-cpu0,
                         'wall_seconds': time.perf_counter()-wall0, 'threads': 1,
                         'lp_solves': 0, 'gpu_used': False}
    return result, coefficients


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=12000)
    parser.add_argument('--test', type=int, default=200000)
    parser.add_argument('--out', type=Path, default=ROOT/'sweep_results.json')
    args = parser.parse_args()
    results, coef = run(args.train, args.test)
    args.out.write_text(json.dumps(results, indent=2)+'\n')
    args.out.with_name(args.out.stem+'_coefficients.json').write_text(json.dumps(coef, indent=2)+'\n')
