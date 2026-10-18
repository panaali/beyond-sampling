"""T223: score tie_blind on leaderboard v3 (m16/leaderboard.py, imported unchanged) without touching the shared files:
RAW / RAW_SNAPSHOT and the report are redirected into this folder. sci_tao's committed raw rows
(results/leaderboard_v3_raw.jsonl) are copied in, so both rows are scored by the same table() code.
Class "quadrature" is assigned here for tie_blind pending DRI review (fixed Sobol node sets, as sci_tao).
usage (from the repo root): python m16/approaches/tie_blind/score.py [run|table] [budgets]  -> m16/approaches/tie_blind/lb/"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "m16"))
import numpy as np
import leaderboard as lb
from leaderboard import fpe_flops, flops
OUT = os.path.join(HERE, "lb"); os.makedirs(OUT, exist_ok=True)
lb.RAW = os.path.join(OUT, "raw.jsonl"); lb.RAW_SNAPSHOT = os.path.join(OUT, "raw_snapshot.jsonl")
lb.CLASS["tie_blind"] = ("quadrature", "fixed scrambled-Sobol faces (non-blind) + fixed Sobol blind points (pending DRI review)")
c_s = flops(16, 10) + 16 * 10 + 3 * 10 - 2          # T196 sampling FLOPs per input (review/cost_convention)


def seed_sci_tao():
    have = set()
    if os.path.exists(lb.RAW):
        have = {(d["name"], d["model"], d["budget"]) for d in map(json.loads, open(lb.RAW))}
    with open(lb.RAW, "a") as f:
        for line in open("results/leaderboard_v3_raw.jsonl"):
            d = json.loads(line)
            if d["name"] == "sci_tao" and d["v"] == lb.VERSION and d["H"] == 16 and d["T"] == 10 and (d["name"], d["model"], d["budget"]) not in have:
                f.write(line)


def report(H=16, T=10, names=None):
    rows = lb.table(H, T, ["sci_tao", "tie_blind"])
    # T196 restatement: estimator FLOPs = 2 x FLOPs per FPE x charged FPE (op counters charged 2x until split, T196);
    # here cost is in forward-pass equivalents on both sides, so the T196 factor is fpe_flops / c_s per side:
    # the ratio is unchanged by the unit except that sampling is charged c_s instead of fpe_flops per input.
    for r in rows:
        r["ratio_T196"] = r["ratio"] * fpe_flops(H, T) / c_s; r["ci_T196"] = [c * fpe_flops(H, T) / c_s for c in r["ci"]]
    json.dump(rows, open(os.path.join(OUT, "table.json"), "w"), indent=1)
    print(f"{'method':10s} {'group':9s} {'budget':>7s} {'cost':>8s} {'ratio':>6s} {'95% CI':>14s} {'bias pp':>8s} {'time':>5s} n")
    for r in sorted(rows, key=lambda r: (r["group"] != "orig", r["group"], r["budget"], r["name"])):
        print(f"{r['name']:10s} {r['group']:9s} {r['budget']:7d} {r['cost']:8.0f} {r['ratio']:6.2f} [{r['ci'][0]:5.2f},{r['ci'][1]:5.2f}] "
              f"{r['bias_pp']:+8.3f} {r['time_ratio']:5.1f} {r['n']}")


lb.report = report
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    budgets = tuple(int(b) for b in (sys.argv[2] if len(sys.argv) > 2 else "1000,4000,16000,64000,256000").split(","))
    seed_sci_tao()
    if cmd == "run":
        lb.run(["tie_blind"], 16, 10, budgets, None)
    else:
        report()
