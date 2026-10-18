"""Markdown summary of results/frontier_<panel>.json: per family the best all-deductive and the best overall setting
(lowest F_max), plus the (C, F_max) Pareto sets.  usage: python summary.py [lb3|dev2] -> results/summary_<panel>.md"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402
from frontier import pareto  # noqa: E402


def main(panel="lb3"):
    t = json.load(open(os.path.join(cf.HERE, "results", f"frontier_{panel}.json")))
    groups = cf.GROUPS if panel == "lb3" else cf.DEV2_GROUPS
    L = [f"| family | setting | label | C (FPE) | F_max | " + " | ".join(f"{g} bias/RMS pp, F" for g in groups) + " |",
         "|---" * (5 + len(groups)) + "|"]

    def row(r):
        return (f"| {r['fam']} | {r['cfg']} | {r['label']} | {r['C']:.0f} | {r['Fmax']:.1f} | " + " | ".join(
            f"{r['groups'][g]['bias_pp']:+.1f}/{r['groups'][g]['rms_pp']:.1f}, {r['groups'][g]['F']:.1f}" for g in groups) + " |")
    for fam in sorted({r["fam"] for r in t}):
        rs = [r for r in t if r["fam"] == fam]
        best = [min(rs, key=lambda r: r["Fmax"])]
        d = [r for r in rs if r["label"] == "D"]
        if d and min(d, key=lambda r: r["Fmax"]) is not best[0]:
            best.insert(0, min(d, key=lambda r: r["Fmax"]))
        L += [row(r) for r in best]
    L += ["", "Pareto set in (C, F_max), all labels: " + "; ".join(
        f"{r['fam']} [{r['cfg']}] C={r['C']:.0f} F_max={r['Fmax']:.1f}" for r in pareto(t, "C", "Fmax")),
          "", "Pareto set in (C, F_max), all-deductive: " + "; ".join(
        f"{r['fam']} [{r['cfg']}] C={r['C']:.0f} F_max={r['Fmax']:.1f}" for r in pareto([r for r in t if r['label'] == 'D'], "C", "Fmax"))]
    L += ["", "Per-group Pareto sets in (C, RMS), all labels:"]
    for g in groups:
        pts = [dict(r, rms=r["groups"][g]["rms_pp"]) for r in t]
        L.append(f"- {g}: " + "; ".join(f"{r['fam']} [{r['cfg']}] C={r['C']:.0f} RMS={r['rms']:.2f} F={r['groups'][g]['F']:.1f}"
                                        for r in pareto(pts, "C", "rms")))
    open(os.path.join(cf.HERE, "results", f"summary_{panel}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lb3")
