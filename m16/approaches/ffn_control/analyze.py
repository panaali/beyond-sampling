"""Accuracy, reference truth and linear-region statistics for the T345 MLPs (and M_16,10 for comparison).

usage:
  python analyze.py acc                 # accuracy of every models/*.json on 1e7 inputs (seed 20261018) -> results/acc.json
  python analyze.py regions             # region statistics from 1e7 inputs (seed 20261019)  -> results/regions.json
  python analyze.py truth NAME N        # reference accuracy of panel model NAME on N inputs (seed 777) -> results/truth_NAME.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.special import comb

sys.path.insert(0, os.path.dirname(__file__))
import walks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T, CHUNK = 10, 500_000


def load(path):
    return walks.MLP(json.load(open(path))["weights"])


def panel():
    """{name: model}: per architecture the 3 seeds plus 9 noisy copies (sigma .05 x RMS per matrix); M_16,10 + 9 copies."""
    sys.path.insert(0, os.path.join(HERE, "../.."))
    import common as c
    out = {}
    for arch in ["L1h22", "L2h13", "L1h43", "L2h21"]:
        paths = sorted(glob.glob(f"{HERE}/models/{arch}_s*.json"))
        if not paths:
            continue
        base = [json.load(open(p))["weights"] for p in paths]
        for p, w in zip(paths, base):
            out[os.path.basename(p)[:-5]] = walks.MLP(w)
        rng = np.random.default_rng(345)
        for i in range(9):
            w = [np.asarray(m) for m in base[i % len(base)]]
            out[f"{arch}_n{i}"] = walks.MLP([m + .05 * np.sqrt((m ** 2).mean()) * rng.standard_normal(m.shape) for m in w])
    out["M16_10"] = walks.RNN(c.Wi, c.Wh, c.Wo)
    rng = np.random.default_rng(345)
    for i in range(9):
        out[f"M16_10_n{i}"] = walks.RNN(*(m + .05 * np.sqrt((m ** 2).mean()) * rng.standard_normal(m.shape)
                                         for m in (c.Wi, c.Wh, c.Wo)))
    return out


def accuracy(model, n, seed):
    rng, ok = np.random.default_rng(seed), 0
    for _ in range(n // CHUNK):
        x = rng.standard_normal((CHUNK, T))
        ok += int((model.forward(x).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    return ok / n


def patterns(model, x):
    """Per-layer cumulative gate patterns as hashable int64 keys (layer 1, layers 1-2, ...)."""
    keys, h, acc = [], x, np.zeros(len(x), dtype=np.uint64)
    for w in model.W[:-1]:
        z = h @ w.T
        g = z > 0
        bits = (g.astype(np.uint64) << np.arange(g.shape[1], dtype=np.uint64)).sum(1, dtype=np.uint64)
        acc = acc * np.uint64(1_000_003) ^ bits ^ (acc >> np.uint64(29))
        keys.append(acc.copy())
        h = np.where(g, z, 0)
    return keys


def rnn_patterns(model, x):
    """Cumulative gate-pattern keys of the RNN after each step k = 1..10 (prefix regions)."""
    keys, h, acc = [], np.zeros((len(x), len(model.Wi))), np.zeros(len(x), dtype=np.uint64)
    for t in range(T):
        z = h @ model.Wh.T + x[:, t:t + 1] * model.Wi[None]
        g = z > 0
        bits = (g.astype(np.uint64) << np.arange(g.shape[1], dtype=np.uint64)).sum(1, dtype=np.uint64)
        acc = acc * np.uint64(1_000_003) ^ bits ^ (acc >> np.uint64(29))
        keys.append(acc.copy())
        h = np.where(g, z, 0)
    return keys


def cell_key(k, pred, tgt):
    """Decision cell = (full gate pattern, predicted class, target class)."""
    return k * np.uint64(100) + pred.astype(np.uint64) * np.uint64(10) + tgt.astype(np.uint64)


def coverage(counts, n):
    """N_delta: fewest regions whose sample mass reaches 1 - delta."""
    c = np.sort(counts)[::-1].cumsum() / n
    return {f"N_{d:.0e}": int(np.searchsorted(c, 1 - d) + 1) for d in (1e-2, 1e-3, 1e-4, 1e-5)}


def main():
    os.makedirs(f"{HERE}/results", exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "acc":
        res = {}
        for p in sorted(glob.glob(f"{HERE}/models/*.json")):
            d = json.load(open(p))
            a = accuracy(walks.MLP(d["weights"]), 10_000_000, 20261018)
            res[os.path.basename(p)[:-5]] = {"arch": d["arch"], "seed": d["seed"], "n_params": d["n_params"],
                                             "acc": a, "se": (a * (1 - a) / 1e7) ** .5,
                                             "final_train_loss": d["log"][-1][1]}
            print(p, res[os.path.basename(p)[:-5]], flush=True)
        json.dump(res, open(f"{HERE}/results/acc.json", "w"), indent=1)
    elif cmd == "regions":
        res, n = {}, 10_000_000
        for p in sorted(glob.glob(f"{HERE}/models/*.json")):
            m = load(p)
            per_layer = None
            rng = np.random.default_rng(20261019)
            allkeys = [[] for _ in m.W[:-1]]
            cells = []
            for _ in range(n // CHUNK):
                x = rng.standard_normal((CHUNK, T))
                for li, k in enumerate(patterns(m, x)):
                    allkeys[li].append(k)
                cells.append(cell_key(k, m.forward(x).argmax(1), np.argsort(x, 1)[:, -2]))
            per_layer = []
            for li, ks in enumerate(allkeys):
                _, cnt = np.unique(np.concatenate(ks), return_counts=True)
                per_layer.append({"distinct_in_1e7": int(len(cnt)), "singletons": int((cnt == 1).sum()),
                                  **coverage(cnt, n)})
            _, cnt = np.unique(np.concatenate(cells), return_counts=True)
            per_layer.append({"decision_cells_in_1e7": int(len(cnt)), **coverage(cnt, n)})
            h = m.W[0].shape[0]
            name = os.path.basename(p)[:-5]
            res[name] = {"layer1_exact_general_position": int(2 * sum(comb(h - 1, i, exact=True) for i in range(T))),
                         "per_layer": per_layer}
            print(name, res[name], flush=True)
        m, rng, allkeys = panel()["M16_10"], np.random.default_rng(20261019), [[] for _ in range(T)]
        cells = []
        for _ in range(n // CHUNK):
            x = rng.standard_normal((CHUNK, T))
            for li, k in enumerate(rnn_patterns(m, x)):
                allkeys[li].append(k)
            cells.append(cell_key(k, m.forward(x).argmax(1), np.argsort(x, 1)[:, -2]))
        res["M16_10"] = {"per_step": []}
        for ks in allkeys:
            _, cnt = np.unique(np.concatenate(ks), return_counts=True)
            res["M16_10"]["per_step"].append({"distinct_in_1e7": int(len(cnt)), "singletons": int((cnt == 1).sum()),
                                              **coverage(cnt, n)})
        _, cnt = np.unique(np.concatenate(cells), return_counts=True)
        res["M16_10"]["per_step"].append({"decision_cells_in_1e7": int(len(cnt)), **coverage(cnt, n)})
        print("M16_10", res["M16_10"], flush=True)
        json.dump(res, open(f"{HERE}/results/regions.json", "w"), indent=1)
    elif cmd == "truth":
        name, n = sys.argv[2], int(float(sys.argv[3]))
        a = accuracy(panel()[name], n, 777)
        json.dump({"name": name, "n": n, "acc": a, "se": (a * (1 - a) / n) ** .5},
                  open(f"{HERE}/results/truth_{name}.json", "w"))
        print(name, a)


if __name__ == "__main__":
    main()
