"""Accuracy, reference truth and linear-region statistics for the T345 MLPs (and M_16,10 for comparison).

usage:
  python analyze.py acc                 # accuracy of every models/*.json on 1e7 inputs (seed 20261018) -> results/acc.json
  python analyze.py regions [NAMES]     # region statistics from 1e7 inputs (seed 20261019)  -> results/regions[_NAMES].json
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


SIGMA = 0.001   # relative weight noise (x RMS of each matrix); Amendment B: .05 destroys M_16,10 (acc .26-.45)


def _noised(ws, rng):
    return [m + SIGMA * np.sqrt((m ** 2).mean()) * rng.standard_normal(m.shape) for m in ws]


def panel(prefixes=None):
    """{name: model}. MLP arch (or ladder config): its trained seeds plus 3 noised copies per seed.
    M16_10: the target, zoo retrains 0-4, and 6 noised copies of the target. Amendment B (2026-10-19)."""
    sys.path.insert(0, os.path.join(HERE, "../.."))
    out = {}
    arches = sorted({os.path.basename(p).split("_s")[0] for p in glob.glob(f"{HERE}/models/*.json")})
    for arch in arches:
        if prefixes is not None and arch not in prefixes:
            continue
        paths = sorted(glob.glob(f"{HERE}/models/{arch}_s*.json"))
        rng = np.random.default_rng(345)
        for p in paths:
            w = [np.asarray(m) for m in json.load(open(p))["weights"]]
            tag = os.path.basename(p)[:-5]
            out[tag] = walks.MLP(w)
            for i in range(3):
                out[f"{tag}_c{i}"] = walks.MLP(_noised(w, rng))
    if prefixes is None or "M16_10" in prefixes:
        import common as c
        from alg_zoo import zoo_2nd_argmax
        out["M16_10"] = walks.RNN(c.Wi, c.Wh, c.Wo)
        for sd in range(5):
            m = zoo_2nd_argmax(16, 10, seed=sd)
            out[f"M16_10_zoo{sd}"] = walks.RNN(m.rnn.weight_ih_l0.detach().numpy()[:, 0].astype(np.float64),
                                               m.rnn.weight_hh_l0.detach().numpy().astype(np.float64),
                                               m.linear.weight.detach().numpy().astype(np.float64))
        rng = np.random.default_rng(345)
        for i in range(6):
            out[f"M16_10_c{i}"] = walks.RNN(*_noised([c.Wi, c.Wh, c.Wo], rng))
    return out


def accuracy(model, n, seed):
    rng, ok = np.random.default_rng(seed), 0
    for _ in range(n // CHUNK):
        x = rng.standard_normal((CHUNK, T))
        ok += int((model.forward(x).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    return ok / n


_HASH = np.random.default_rng(64).integers(1, 2 ** 63, size=1024, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)


def patterns(model, x):
    """Per-layer cumulative gate patterns as hashable int64 keys (layer 1, layers 1-2, ...)."""
    keys, h, acc = [], x, np.zeros(len(x), dtype=np.uint64)
    for w in model.W[:-1]:
        z = h @ w.T
        g = z > 0
        bits = np.concatenate([(g[i:i + 100_000].astype(np.uint64) * _HASH[:g.shape[1]]).sum(1, dtype=np.uint64)
                               for i in range(0, len(g), 100_000)])   # random 64-bit hash (any width), sliced for memory
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
        names = sys.argv[2].split(",") if len(sys.argv) > 2 else None   # optional subset, e.g. D3w32_s0_n28,M16_10
        out_path = f"{HERE}/results/regions{'' if names is None else '_' + sys.argv[2].replace(',', '+')}.json"
        for p in sorted(glob.glob(f"{HERE}/models/*.json")):
            if names is not None and os.path.basename(p)[:-5] not in names:
                continue
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
            row_sum = np.abs(m.W[0].sum(1)) / np.linalg.norm(m.W[0], axis=1)   # ~0: near shift-invariance
            res[name] = {"layer1_general_position_bound": int(2 * sum(comb(h - 1, i, exact=True) for i in range(T))),
                         "layer1_max_rowsum_over_norm": float(row_sum.max()),
                         "good_turing_unseen_mass": [round(d.get("singletons", 0) / n, 4) for d in per_layer[:-1]],
                         "per_layer": per_layer}
            print(name, res[name], flush=True)
        if names is not None and "M16_10" not in names:
            json.dump(res, open(out_path, "w"), indent=1)
            return
        m, rng, allkeys = panel(["M16_10"])["M16_10"], np.random.default_rng(20261019), [[] for _ in range(T)]
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
        json.dump(res, open(out_path, "w"), indent=1)
    elif cmd == "truth":
        name, n = sys.argv[2], int(float(sys.argv[3]))
        a = accuracy(panel([name.split("_s")[0].split("_zoo")[0].split("_c")[0]])[name], n, 777)
        json.dump({"name": name, "n": n, "acc": a, "se": (a * (1 - a) / n) ** .5},
                  open(f"{HERE}/results/truth_{name}.json", "w"))
        print(name, a)


if __name__ == "__main__":
    main()
