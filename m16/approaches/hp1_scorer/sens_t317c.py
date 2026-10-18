"""T281 sensitivity for T317c: charge every executed ndtr in cont_c.profile (np.where evaluates Phi for every merged
interval at both ends, masked values included) on top of the official cost (raw_t317c_cont_off). Asserts that the
official cost reproduces. usage: PANEL=hp2|hp3 PYTHONPATH=shim python sens_t317c.py"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score  # noqa: E402
f = score.load_sub(os.path.join(score.ROOT, "data/subs/t317c_1c742430/m16/approaches/r1_cont/estimator_c.py"), "est")
import cont_c, walk as W  # noqa: E402
orig, oci = cont_c.profile, W.correct_interval
ext, cnt = {"f": 0.0}, {"rows": 0}


def prof(R, A, B, n, nodes, kind, tmax, phi0):
    out, fc = orig(R, A, B, n, nodes, kind, tmax, phi0)
    ext["f"] += len(nodes) * (2 * len(R) * (20 + (3 if kind == "23" else 0)) + 2 * len(R)) - fc
    return out, fc


def ci(*a):
    cnt["rows"] += len(a[0])
    return oci(*a)


cont_c.profile, W.correct_interval = prof, ci
names, man, models, tr = score.load_panel()
raw = {r["k"]: r for r in map(json.loads, open(os.path.join(score.D, "raw_t317c_cont_off.jsonl")))}
rows = []
for k, nm in enumerate(names):
    ext["f"] = 0.0; cnt["rows"] = 0
    e, c = f(models[k], None)
    assert e == raw[k]["est"] and abs(c + 13 * cnt["rows"] - raw[k]["cost"]) < 1e-6, k
    rows.append(dict(k=k, name=nm, B=0, est=e, cost=raw[k]["cost"] + ext["f"]))
with open(os.path.join(score.D, "raw_t317c_cont_allphi.jsonl"), "w") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in rows)
score.aggregate_F("t317c_cont_allphi", "deductive", (0,))
