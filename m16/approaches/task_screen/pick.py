"""Apply the pre-registered pick rule v2 (README, amendment 1) to the stage-B accuracies.

usage: python pick.py      reads data/task_screen/acc_30.json, prints the markdown table and the picks,
                           writes results/acc_30.json and results/picks.json
"""
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../../../data/task_screen")

KTH = ["argmax", "3rd_argmax", "4th_argmax", "5th_argmax", "2nd_argmin"]
RANKABS = ["argmax_abs", "2nd_argmax_abs", "argmin_abs"]
ADJ = ["argmax_absdiff", "argmin_absdiff", "argmax_diff", "argmax_pairsum", "argmax_window3", "argmax_absdiff_lag2",
       "argmax_prod", "argmax_peak"]
RUNSUM = ["argmax_cumsum", "argmax_cumsum_abs", "argmax_suffixsum"]
COUNTS = ["n_records", "n_sign_changes", "n_ascents"]
PAIRS = ["pair_top2", "pair_extremes", "pair_top2_abs", "pair_bottom2_abs", "pair_closest"]
D = {"2nd_argmin": 0, "pair_top2": 1, "pair_extremes": 2, "pair_top2_abs": 2, "pair_bottom2_abs": 2, "pair_closest": 3,
     "argoutlier": 4}
D.update({t: 1 for t in KTH[:4]}, **{t: 2 for t in RANKABS}, **{t: 3 for t in ADJ}, **{t: 4 for t in RUNSUM + COUNTS})
SOFT = [(RANKABS, 2), (ADJ, 3), (RUNSUM, 2), (COUNTS, 2)]


def hard_ok(t, picked):
    if t in PAIRS and sum(p in PAIRS for p in picked) >= 3:
        return False
    if t in KTH and sum(p in KTH for p in picked) >= 4:
        return False
    return True


def soft_ok(t, picked):
    return all(not (t in fam and sum(p in fam for p in picked) >= cap) for fam, cap in SOFT)


def main():
    r = json.load(open(os.path.join(ROOT, "acc_30.json")))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "acc_30.json"), os.path.join(HERE, "results", "acc_30.json"))
    by = {}
    for k, v in r.items():
        seed = int(k.rsplit("_", 1)[1])
        by.setdefault(v["task"], {})[seed] = v["acc"]
    rows = {t: {"seeds": s, "mean": float(np.mean(list(s.values()))), "best_seed": max(s, key=s.get)} for t, s in by.items()}
    elig = [t for t in rows if rows[t]["mean"] >= 0.90 and t not in ("2nd_argmax", "2nd_argmin")]
    order = sorted(elig, key=lambda t: (-D[t], abs(rows[t]["mean"] - 0.95)))
    picks = []
    for t in order:
        if len(picks) < 10 and hard_ok(t, picks) and soft_ok(t, picks):
            picks.append(t)
    for t in sorted(elig, key=lambda t: abs(rows[t]["mean"] - 0.95)):
        if len(picks) < 10 and t not in picks and hard_ok(t, picks):
            picks.append(t)
    if len(picks) < 10 and "2nd_argmin" in rows and rows["2nd_argmin"]["mean"] >= 0.90:
        picks.append("2nd_argmin")
    picks = sorted(picks, key=lambda t: (-D[t], abs(rows[t]["mean"] - 0.95)))
    near = sorted([t for t in rows if rows[t]["mean"] < 0.90], key=lambda t: -rows[t]["mean"])

    print("| task | D | seed 0 | seed 1 | seed 2 | mean | picked |\n|---|---|---|---|---|---|---|")
    for t in sorted(rows, key=lambda t: -rows[t]["mean"]):
        s = rows[t]["seeds"]
        cells = " | ".join(f"{s[i]:.4f}" if i in s else "-" for i in range(3))
        mark = f"**#{picks.index(t) + 1}**" if t in picks else ""
        print(f"| `{t}` | {D.get(t, '-')} | {cells} | {rows[t]['mean']:.4f} | {mark} |")
    print("\npicks:", picks, "\nnear misses:", [(t, round(rows[t]["mean"], 4)) for t in near])
    json.dump({"picks": [{"task": t, "D": D[t], "mean": rows[t]["mean"], "primary_seed": rows[t]["best_seed"],
                          "seeds": rows[t]["seeds"]} for t in picks],
               "near_misses": {t: rows[t]["mean"] for t in near}, "all": rows},
              open(os.path.join(HERE, "results", "picks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
