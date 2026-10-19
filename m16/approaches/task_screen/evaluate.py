"""Accuracy of every trained T419 model by Monte Carlo in numpy float64.

usage: python evaluate.py log2_n_train [n_samples]     (default n = 10^7, chunks of 2^18, input rng seed 419)
Same inputs for every model (common random numbers). Appends/updates data/task_screen/acc_<log2n>.json and prints a table.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import is_pair, labels, pair_labels  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/task_screen"))


def logits(x, Wi, Wh, Wo):
    h = np.zeros((len(x), len(Wi)))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def main(log2n, n=10**7, chunk=2**18):
    files = sorted(glob.glob(os.path.join(ROOT, f"*_{log2n}_*.npz")))
    out = os.path.join(ROOT, f"acc_{log2n}.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    todo = [f for f in files if os.path.basename(f)[:-4] not in res or res[os.path.basename(f)[:-4]]["n"] != n]
    models = {os.path.basename(f)[:-4]: np.load(f) for f in todo}
    hits = {k: 0 for k in models}
    rng = np.random.default_rng(419)
    done = 0
    lab = {}
    while done < n:
        m = min(chunk, n - done)
        x = rng.standard_normal((m, 10))
        lab.clear()
        for k, w in models.items():
            task = str(w["task"])
            if task not in lab:
                lab[task] = pair_labels(task, x) if is_pair(task) else labels(task, x)
            z = logits(x, w["Wi"], w["Wh"], w["Wo"])
            if is_pair(task):   # readout: the two largest logits as an unordered set; exact set match
                hits[k] += int((np.sort(np.argsort(z, 1)[:, -2:], 1) == lab[task]).all(1).sum())
            else:
                hits[k] += int((z.argmax(1) == lab[task]).sum())
        done += m
    for k in models:
        res[k] = {"task": str(models[k]["task"]), "acc": hits[k] / n, "n": n}
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    for k in sorted(res):
        print(f"{k:32s} {res[k]['acc']:.5f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), *map(int, sys.argv[2:3]))
