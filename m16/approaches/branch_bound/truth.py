"""Ground truth for noisy_models(3, 0.0025), n=4M (arc_eval.true_acc). Evaluation only, not part of the method."""
import sys, json
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import true_acc, noisy_models
out = [float(true_acc(m, n=4_000_000)) for m in noisy_models(3, 0.0025)]
print(out)
json.dump({"0.0025": out, "orig": 0.95317945}, open("truth.json", "w"), indent=1)
