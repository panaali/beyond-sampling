"""Ground truth for noisy_models(6, sigma), n=4M (arc_eval.true_acc, seed 12345). Cached to truth.json."""
import sys, json, os
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import true_acc, noisy_models
HERE = os.path.dirname(os.path.abspath(__file__))
out = {"0": [0.95317945]}
for s in (0.0025, 0.005):
    out[str(s)] = [float(true_acc(m, n=4_000_000)) for m in noisy_models(6, s)]
    print(s, out[str(s)], flush=True)
json.dump(out, open(os.path.join(HERE, "truth.json"), "w"), indent=1)
