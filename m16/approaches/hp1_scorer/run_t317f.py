"""T281: score T317f (r1_cont/estimator_f.py @ 88453669) once per panel, recording per model the estimate, the
as-executed FLOPs (official), and the number of correct_interval row-calls for an audit bound on the screen charge:
the code charges 6C+2 per screen try; my itemization is ~8C+2 plus 2C for the uncharged L update of screened rows,
i.e. <= 40 FLOPs/try more; tries <= 2 x CI rows + 3 x 720 (start of each line). usage: PANEL=hp2|hp3|hp4 python run_t317f.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score  # noqa: E402
est = score.load_sub(os.path.join(score.ROOT, "data/subs/t317f_88453669/m16/approaches/r1_cont/estimator_f.py"), "est")
import cont_f  # noqa: E402
oci = cont_f.correct_interval; cnt = {"r": 0}


def ci(*a):
    cnt["r"] += len(a[0])
    return oci(*a)


cont_f.correct_interval = ci
names, man, models, tr = score.load_panel()
path = os.path.join(score.D, "raw_t317f_cont.jsonl")
assert not os.path.exists(path), "already scored on this panel"
rows = []
for k, nm in enumerate(names):
    cnt["r"] = 0
    e, c = est(models[k], None)
    rows.append(dict(k=k, name=nm, B=0, est=e, cost=c, ci_rows=cnt["r"]))
with open(path, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
for tag, mult, add in (("t317f_cont", 1, 0), ("t317f_cont_stack", 1.09 * 1.04, 0), ("t317f_cont_screenbound", 1, 1)):
    if tag != "t317f_cont":
        with open(os.path.join(score.D, f"raw_{tag}.jsonl"), "w") as f:
            for r in rows:
                r2 = dict(r, cost=r["cost"] * mult + add * 40 * (2 * r["ci_rows"] + 3 * 720))
                f.write(json.dumps(r2) + "\n")
    score.aggregate_F(tag, "deductive", (0,))
