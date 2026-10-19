"""T391 development: var*cost over 16 alternative rules, frozen vs isg, on held-out DEV2 models (40-79 subset)."""
import importlib.util
import numpy as np
D = np.load("../../../data/dev2/weights.npz")
def load(path, nm):
    spec = importlib.util.spec_from_file_location(nm, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E0, E1 = load("estimator.py", "e0"), load("isg/estimator.py", "e1")
for i in (41, 47, 53, 59, 65, 71):
    m = (D["Wi"][i], D["Wh"][i], D["Wo"][i]); out = []
    for b in (16000, 64000):
        r0 = [E0.est(m, b, rule_seed=9000 + 31 * k) for k in range(16)]
        r1 = [E1.est(m, b, rule_seed=9000 + 31 * k) for k in range(16)]
        v = lambda r: np.var([e for e, _ in r]) * np.mean([c for _, c in r])
        out.append(f"{b}: frozen/isg var*cost {v(r0)/v(r1):.2f}")
    print(f"DEV2[{i}] " + " | ".join(out), flush=True)
