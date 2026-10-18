"""Run the unchanged LB3 harness on probe_v5a / probe_v5b. Deterministic methods are registered in memory as class
'quadrature' (run once per model and budget after LB3's own determinism check). Raw rows go to this folder.
Usage (from the LB3 checkout root): PYTHONPATH=<alg_zoo shim> python <this> <lb3_root>"""
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1]
os.chdir(root)
sys.path.insert(0, os.path.join(root, "m16"))
import leaderboard as L
for name, fn, doc in (("probe_v5a", "est", "exact-law top-3 Sobol nodes + canonical lower ranks"),
                      ("probe_v5b", "est_spread", "v5a nodes + weight-derived lower-rank spread"),
                      ("probe_v5q", "est_full", "control: full 10-D exact-law Sobol nodes")):
    d = os.path.join(root, "m16/approaches", name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "estimator.py"), "w") as f:
        f.write(f"import sys\nsys.path.insert(0, {here!r})\nimport v5\n"
                f"def est(model, budget, rng):\n    return v5.{fn}(model, budget, rng)\n")
    L.CLASS[name] = ("quadrature", doc)
L.RAW = os.path.join(here, "results/lb3_raw.jsonl")
L.RAW_SNAPSHOT = os.path.join(here, "results/lb3_raw_snapshot.jsonl")
L.run(sys.argv[2].split(","), 16, 10, list(L.CURVE_BUDGETS), None)
