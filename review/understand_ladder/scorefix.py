"""T97: re-score all score.py models with the id()-reuse fix (fibre_fix), LOOK = 2 (F2L2) and 3.
usage: OMP_NUM_THREADS=1 python scorefix.py H  -> score_fix_H{H}.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import score  # noqa: E402
import fibre_fix  # noqa: E402

score.fibre = fibre_fix
H = int(sys.argv[1])
rows = score.run(H, [2, 3])
summ = {f"L{L}": score.summarize(H, rows, f"L{L}") for L in (2, 3)}
json.dump({"H": H, "rows": rows, "summary": summ}, open(os.path.join(HERE, f"score_fix_H{H}.json"), "w"), indent=1)
for k, S in summ.items():
    for g, d in S.items():
        print(f"H={H} fixed {k} {g:12s} n={d['n']:3d} rmse {d['rmse']:.2e} max_over {d['max_over']:+.2e} min_err? ops {d['ops']:.0f} "
              f"group {d['group_ratio']:.2e} worst {d['worst_model_ratio']:.2e} ({d['worst']}) fails {d['n_fail']}")
