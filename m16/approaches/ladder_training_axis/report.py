"""T101: k-ladder tables from rung_k*.json, mech_k*.json. usage: python report.py  -> prints the markdown tables"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def f(x):
    return "0" if x == 0 else (f"{x:.2g}" if x < 0.01 else f"{x:.3f}")


def main():
    lines = []
    rungs = sorted((int(re.findall(r"k(\d+)", p)[0]) for p in glob.glob(os.path.join(HERE, "rung_k*.json"))), reverse=True)
    lines.append("| k (n_seqs=2^k) | clean acc, seeds 0-4 | regions clean (panel mean) | panel acc | exact ops | T75 ops | **T75 panel MSE ratio, seeds 0-4** | T75 worst CI hi | F2L3 ops | **F2L3 ratio, seeds 0-4** | T75 / F2L3 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in rungs:
        r = json.load(open(os.path.join(HERE, f"rung_k{k}.json")))
        S = [r["seeds"][str(s)] for s in range(5) if str(s) in r["seeds"]]
        if len(S) < 5:
            lines.append(f"| {k} | (running: {len(S)}/5 seeds) |"); continue
        acc = " / ".join(f"{d['clean']['truth']:.3f}" for d in S)
        reg = f"{np.mean([d['clean']['regions'] for d in S]):.0f} ({np.mean([d['panel']['regions_mean'] for d in S]):.0f})"
        pacc = f"{np.mean([d['panel']['mean_truth'] for d in S]):.3f}"
        eops = f"{np.mean([d['panel']['exact_ops'] for d in S])/1e3:.1f}k"
        tops = f"{np.mean([d['panel']['t75']['ops'] for d in S])/1e3:.1f}k"
        tr = " / ".join(f(d["panel"]["t75"]["ratio"]) for d in S)
        thi = f(max(d["panel"]["t75"]["ci"][1] for d in S))
        fops = f"{np.mean([d['panel']['f2l3']['ops'] for d in S])/1e3:.0f}k"
        fr = " / ".join(f(d["panel"]["f2l3"]["ratio"]) for d in S)
        v = []
        for nm in ("t75", "f2l3"):
            ok = all(d["panel"][nm]["ratio"] < 1 and d["clean"][nm + "_ratio"] < 1 for d in S)
            v.append("pass" if ok else "**FAIL**")
        lines.append(f"| {k} | {acc} | {reg} | {pacc} | {eops} | {tops} | {tr} | {thi} | {fops} | {fr} | {' / '.join(v)} |")
    lines.append("")
    lines.append("Mechanism (clean models, `mech.py`): min over neurons of |r - 1| for the step-2 switch ratio r = x1/x0, chart x0>0 / x0<0, seeds 0-4; dead-cell error share")
    lines.append("")
    lines.append("| k | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | dead (all-off) error share, seeds 0-4 |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in sorted(glob.glob(os.path.join(HERE, "mech_k*.json")), key=lambda p: -int(re.findall(r"k(\d+)", p)[0])):
        k = int(re.findall(r"k(\d+)", p)[0]); M = json.load(open(p))
        cells = []
        for s in range(5):
            ch = M[str(s)]["charts"]
            cells.append(f"{ch['1']['near1']:.2f} / {ch['-1']['near1']:.2f}")
        dead = " / ".join(f"{M[str(s)]['dead_share']:.0%}" for s in range(5))
        lines.append(f"| {k} | " + " | ".join(cells) + f" | {dead} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
