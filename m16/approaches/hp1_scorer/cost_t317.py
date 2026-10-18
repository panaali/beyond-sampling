"""T281 official cost for the r1_cont family (DRI ruling 2026-10-18): the code's executed count, with the hand-entered
correct_interval constant (146 FLOPs per row-call) replaced by the full itemized count of walk.correct_interval for
C = 10 logits (every arithmetic, comparison and boolean op):
  c = (L_k - L) - (dL_k - dL) tc : 40 | e : 10 | root = -c/e : 20 | lb/ub masks : 20 |
  bad0 (e==0, c<0, |c|, |L|, max, 1+, *1e-12, <=, jj<k, &, |, &) : 102 | lb.max, ub.min, maximum, minimum : 22 |
  ok (>, any, ~, &) : 13 | final where x2 : 2                                   = 229 FLOPs per row-call.
usage: PANEL=hp1|hp2 python cost_t317.py <name> <estimator file>"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score  # noqa: E402
CI_FULL, CI_CODE = 229, 146
name, path = sys.argv[1], sys.argv[2]
est = score.load_sub(path, "est")
import walk as W  # noqa: E402
orig = W.correct_interval; cnt = {"rows": 0}


def ci(L, dL, tc, k, lo, hi):
    cnt["rows"] += len(L)
    return orig(L, dL, tc, k, lo, hi)


W.correct_interval = ci
names, man, models, tr = score.load_panel()
old = {r["k"]: r for r in map(json.loads, open(os.path.join(score.D, f"raw_{name}.jsonl")))}
rows = []
for k, nm in enumerate(names):
    cnt["rows"] = 0
    e, c = est(models[k], None)
    assert e == old[k]["est"] and c == old[k]["cost"], k
    rows.append(dict(k=k, name=nm, B=0, est=e, cost=c + (CI_FULL - CI_CODE) * cnt["rows"], cost_code=c,
                     ci_rows=cnt["rows"]))
with open(os.path.join(score.D, f"raw_{name}_off.jsonl"), "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
c0 = np.mean([r["cost_code"] for r in rows]); c1 = np.mean([r["cost"] for r in rows])
print(f"{name} {score.PANEL}: code {c0 / score.FPE:.0f} FPE -> official {c1 / score.FPE:.0f} FPE (x{c1 / c0:.4f})")
score.aggregate_F(name + "_off", "deductive", (0,))
