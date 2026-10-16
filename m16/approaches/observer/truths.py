"""Ground truth for the noisy copies (n=4,000,000 each, arc_eval.true_acc)."""
import sys, json
sys.path.insert(0, '/home/user/clean-neurons/m16')
from arc_eval import noisy_models, true_acc
out = {}
for sig in (0.0025, 0.005):
    out[str(sig)] = [float(true_acc(m, n=4_000_000)) for m in noisy_models(6, sig)]
    print(sig, out[str(sig)], flush=True)
json.dump(out, open('truths.json', 'w'), indent=1)
