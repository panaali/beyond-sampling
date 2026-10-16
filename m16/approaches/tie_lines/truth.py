"""Ground truth (n=4,000,000) for noisy_models(6, 0.0025) and (6, 0.005); cached to truth.json."""
import sys, json
sys.path.insert(0, "m16")
from arc_eval import noisy_models, true_acc
out = {}
for sig in (0.0025, 0.005):
    out[str(sig)] = [float(true_acc(m, n=4_000_000)) for m in noisy_models(6, sig)]
    print(sig, out[str(sig)], flush=True)
json.dump(out, open("m16/approaches/tie_lines/truth.json", "w"), indent=1)
