"""Independent fixed-size validation reference; never estimator training data."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent

def main():
    import torch
    start = time.process_time()
    checkpoint = Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    wi, wh, wo = (state[k].numpy().astype(np.float64) for k in
                  ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
    wi = wi.ravel()
    rng = np.random.default_rng(52010)
    total = 2**20
    correct = 0
    label_counts = np.zeros(10, dtype=np.int64)
    for _ in range(total // 65536):
        x = rng.standard_normal((65536, 10))
        labels = np.argsort(x, axis=1)[:, -2]
        h = np.zeros((len(x), 2))
        for t in range(10):
            h = np.maximum(h @ wh.T + x[:, t, None]*wi, 0)
        correct += int(np.sum(np.argmax(h @ wo.T, axis=1) == labels))
        label_counts += np.bincount(labels, minlength=10)
    p = correct/total
    se = np.sqrt(p*(1-p)/total)
    report = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  seed=52010, samples=total, chunk=65536, correct=correct,
                  accuracy=p, standard_error=se, normal_95_interval=[p-1.96*se, p+1.96*se],
                  label_counts=label_counts.tolist(), dtype='float64',
                  cpu_seconds=time.process_time()-start,
                  purpose='Independent validation of frozen T50 grid9 outputs; no fitting or winner selection',
                  charged_full_forward_evaluations=total)
    (HERE/'t10_reference.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
