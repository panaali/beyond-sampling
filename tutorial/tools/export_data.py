"""Export the data the tutorial page needs (T374).

Writes tutorial/data/weights.json (real AlgZoo weights, downloaded from ARC's public bucket, the same files
alg_zoo.zoo_2nd_argmax / example_2nd_argmax load) and tutorial/data/panels.json (DEV3 truths and the T234 training
ceiling summary, copied from the T192 branch). Every block carries a "source" field.

Usage (CPU, < 1 minute, < 300 MB):
  python tutorial/tools/export_data.py --cache <dir with .pth files or empty dir> \
      --dev3 <dev3/truths.json> --grid3 <capacity_ceiling/results/grid3_summary.json>
The last two come from `git show origin/claude/t192-gpu-cost-law:m16/approaches/...`.
"""
import argparse, json, os, urllib.request
import numpy as np
import torch

BUCKET = "https://storage.googleapis.com/arc-ml-public/alg"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch(cache, name, url):
    p = os.path.join(cache, name)
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    sd = torch.load(p, weights_only=True, map_location="cpu")
    return {"Wi": sd["rnn.weight_ih_l0"][:, 0].double().numpy(),
            "Wh": sd["rnn.weight_hh_l0"].double().numpy(),
            "Wo": sd["linear.weight"].double().numpy(), "url": url}


def acc(m, n=1_000_000, seed=0):
    T = m["Wo"].shape[0]
    x = np.random.default_rng(seed).standard_normal((n, T))
    h = np.zeros((n, len(m["Wi"])))
    for t in range(T):
        h = np.maximum(h @ m["Wh"].T + x[:, t:t + 1] * m["Wi"][None], 0)
    return float(((h @ m["Wo"].T).argmax(1) == np.argsort(x, 1)[:, -2]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dev3", required=True)
    ap.add_argument("--grid3", required=True)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    models = {"M_16,10": fetch(a.cache, "target.pth", f"{BUCKET}/one_layer_16_hidden_94_acc_2nd_argmax.pth")}
    for H, T in [(2, 3), (4, 3), (8, 3), (16, 3), (16, 4), (16, 5), (16, 10)]:
        for s in range(5) if (H, T) == (16, 10) else [0]:
            nm = f"zoo({H},{T},seed {s})"
            models[nm] = fetch(a.cache, f"zoo_{H}_{T}_{s}.pth", f"{BUCKET}/zoo/2nd_argmax_{H}_{T}_1073741824_{s}.pth")
    out = {"source": "ARC public AlgZoo bucket (gs://arc-ml-public/alg), fully trained (2^30 sequences); quick_acc = "
                     "this script's 1e6-sample check (seed 0), not an answer key", "models": {}}
    for nm, m in models.items():
        out["models"][nm] = {"H": int(len(m["Wi"])), "T": int(m["Wo"].shape[0]), "url": m["url"],
                             "quick_acc": round(acc(m), 4),
                             "Wi": [round(float(v), 7) for v in m["Wi"]],
                             "Wh": [[round(float(v), 7) for v in r] for r in m["Wh"]],
                             "Wo": [[round(float(v), 7) for v in r] for r in m["Wo"]]}
        print(nm, out["models"][nm]["quick_acc"])
    json.dump(out, open(os.path.join(OUT, "weights.json"), "w"), separators=(",", ":"))

    dev3 = json.load(open(a.dev3))["truths"]
    g3 = json.load(open(a.grid3))
    panels = {
        "dev3": {"source": "T342, branch claude/t192-gpu-cost-law, m16/approaches/gpu_cost_law/dev3/truths.json "
                           "(1e8 float64 samples per model)",
                 "acc": [round(v["acc"], 5) for v in dev3.values()]},
        "t234": {"source": "T234 (unverified), branch claude/t192-gpu-cost-law f98ef050, "
                           "capacity_ceiling/results/grid3_summary.json",
                 "rows": [{"log2_data": int(k.split("_n")[1]), "mean": round(v["mean"], 4),
                           "median": round(v["median"], 4), "max": round(v["max"], 4), "sd": round(v["sd"], 4),
                           "n": v["n"]} for k, v in g3.items() if k.startswith("zoo_")]},
    }
    json.dump(panels, open(os.path.join(OUT, "panels.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
