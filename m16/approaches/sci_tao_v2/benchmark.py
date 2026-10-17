"""DEV-only estimator smoke, determinism, actual-cost and CPU-time check."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import core as C
from check import checkpoint_model
from estimator import estimate_details, est


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default=str(Path(__file__).with_name('dev_benchmark.json')))
    args = p.parse_args()
    base = checkpoint_model(args.checkpoint)
    records = []
    for sigma in (.0025, .02):
        rng = np.random.default_rng(1)
        model = tuple(w + sigma * np.sqrt((w*w).mean()) * rng.standard_normal(w.shape) for w in base)
        # Low precision truth: diagnostic only, no claim of a competitive MSE curve.
        x = np.random.default_rng(9021).standard_normal((131072, 10))
        start = time.process_time()
        truth = float(np.mean(C.forward(model, x).argmax(axis=1) == np.argsort(x, axis=1)[:, -2]))
        sampling_seconds_per_fpe = (time.process_time() - start) / len(x)
        for budget in (1000, 4000, 16000):
            start = time.process_time()
            accuracy, cost, info = estimate_details(model, budget)
            elapsed = time.process_time() - start
            # RNG is ignored: confirm across two distinct passed generators.
            assert est(model, budget, np.random.default_rng(1)) == est(model, budget, np.random.default_rng(999))
            assert cost >= info['walk_fpe']
            record = {'model': f'dev_s{sigma}_0', 'budget': budget, 'estimate': accuracy,
                      'actual_cost_fpe_proxy': cost, 'cpu_seconds': elapsed,
                      'sampling_cpu_seconds_per_fpe': sampling_seconds_per_fpe,
                      'cpu_time_ratio_at_actual_cost': elapsed / (sampling_seconds_per_fpe * cost),
                      'diagnostic_truth': truth, 'truth_n': len(x),
                      'truth_se': np.sqrt(truth * (1-truth) / len(x)), **info}
            records.append(record)
    Path(args.output).write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
