"""T340: run variants A and B on orig, seed0, seed3 -> results/est.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier"))
import cf  # noqa: E402
import rare  # noqa: E402

ms = cf.lb3_models(); out = {}
for name in ("orig", "seed0", "seed3"):
    for v in ("A", "B"):
        r, f, info = rare.est(ms[name], variant=v)
        out[f"{name}:{v}"] = dict(est={str(k): x for k, x in r.items()}, C=f / cf.FPE, info=info)
        print(name, v, {k: f"{x:.3e}" for k, x in r.items()}, f"C {f / cf.FPE:.0f}", flush=True)
json.dump(out, open(os.path.join(HERE, "results", "est.json"), "w"), indent=1)
