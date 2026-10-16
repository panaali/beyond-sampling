"""LLM-free population statistics over all stored units (896 neurons, 112 random directions), both poles.

- mass_above_20: share of a pole's positive activation mass on tokens >= 20% of its max
  (how much of what the unit does happens where explainers look).
- top_token_share: share of the top-100 activations (one per document) whose marked token is the pole's
  single most common marked token (a crude "one job" proxy that needs no judgement).
- kurtosis: excess kurtosis of the activation distribution (privileged-basis signature; random directions
  in a space with a privileged basis should be closer to Gaussian).
"""
import json
from collections import Counter

import numpy as np
from scipy.stats import kurtosis, mannwhitneyu

A, SKIP = "data/acts", 4
tokens = np.load(f"{A}/tokens.npy")
doc_of = np.load(f"{A}/doc_of.npy")
shape = tuple(json.load(open(f"{A}/meta.json"))["shape"])
n_done = json.load(open(f"{A}/progress.json"))["done"]
acts = np.memmap(f"{A}/acts.f16", dtype=np.float16, mode="r", shape=shape)
N_NEU = np.load(f"{A}/neuron_idx.npy").shape[1]

rows = []
for layer in range(shape[2]):
    block = np.array(acts[:n_done, SKIP:, layer, :], dtype=np.float32)  # (W, P, U)
    for u in range(shape[3]):
        x = block[:, :, u]
        k = float(kurtosis(x.reshape(-1)[::7]))
        for sign in (1, -1):
            v = sign * x
            mx = v.max()
            pos = v[v > 0].sum()
            flat = v.reshape(-1)
            order = np.argpartition(-flat, 3000)[:3000]
            order = order[np.argsort(-flat[order])]
            seen, toks = set(), []
            for f in order:
                w, p = divmod(int(f), v.shape[1])
                if doc_of[w] in seen:
                    continue
                seen.add(doc_of[w]); toks.append(int(tokens[w, p + SKIP]))
                if len(toks) == 100:
                    break
            rows.append({"layer": layer, "kind": "neuron" if u < N_NEU else "random_dir", "sign": sign,
                         "mass_above_20": float(v[v >= 0.2 * mx].sum() / pos),
                         "top_token_share": Counter(toks).most_common(1)[0][1] / len(toks),
                         "kurtosis": k})
    print("layer", layer, flush=True)

out = {}
for m in ("mass_above_20", "top_token_share", "kurtosis"):
    a = [r[m] for r in rows if r["kind"] == "neuron"]
    b = [r[m] for r in rows if r["kind"] == "random_dir"]
    out[m] = {"neuron_median": float(np.median(a)), "random_dir_median": float(np.median(b)),
              "mannwhitney_p": float(mannwhitneyu(a, b).pvalue)}
    out[m]["by_layer_third_neuron"] = [float(np.median([r[m] for r in rows if r["kind"] == "neuron"
                                                        and t * 10 <= r["layer"] < (t + 1) * 10]))
                                       for t in range(3)]
json.dump({"summary": out, "rows": rows}, open("results/population.json", "w"))
print(json.dumps(out, indent=1))
