"""T401 one-page safe-use rating chart + table, from results/t401_est.json (frozen K3) and results/t401_mc.json (conditional MC).
Labels (pre-registered): rated (validated) if MC hits >= 30 and triple-tie share < .5; unrated if tie share >= .5 (unvalidated
points inherit from the nearest validated point of the model); rated (beyond test) if MC hits < 30 at 1e8 and the nearest
validated point is not tie-dominated. -> results/t401_rating.png, results/t401_rating.md"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(HERE, "results", "t401_est.json")))["rows"]
M = json.load(open(os.path.join(HERE, "results", "t401_mc.json")))
MODELS = (("orig", "M_16,10 (the target)"), ("dev3_s529", "retrain dev3_s529"), ("dev3_s531", "retrain dev3_s531"))
T1 = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5); T2 = (0.5, 0.75, 1.0, 1.25, 1.5)
INK, INK2, MUTED, GRID, SURF, BLUE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb", "#2a78d6"


def rows(nm, kind, grid):
    out = []; last_tie = None
    for t in grid:
        mc = M[f"{nm}:{kind}:{t}"]; est_joint = E[nm][kind][str(t)]; est = est_joint / mc["P_cond"]
        validated = mc["hits"] >= 30
        tie = mc["tie_share"]
        if validated:
            if kind == "E1":
                last_tie = tie
            label = "unrated" if (kind == "E1" and tie is not None and tie >= 0.5) else "rated (validated)"
        else:
            label = "unrated" if (last_tie is not None and last_tie >= 0.5) else "rated (beyond test)"
        out.append(dict(t=t, est=est, est_joint=est_joint, mc=mc["rate"] if validated else None, mc_se=mc["rate_se"],
                        hits=mc["hits"], n=mc["n"], tie=tie, label=label, P=mc["P_cond"],
                        ratio=(est / mc["rate"]) if (validated and mc["rate"] > 0) else None,
                        plain_n10=10 / est_joint if est_joint > 0 else float("inf")))
    return out


def panel(ax, rs, title, ylab):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(True, which="major", color=GRID, lw=.8); ax.set_axisbelow(True)
    t = [r["t"] for r in rs]; e = [r["est"] for r in rs]
    ax.plot(t, e, color=BLUE, lw=2, zorder=2)
    for r in rs:
        if r["label"] == "rated (validated)":
            ax.plot(r["t"], r["est"], "o", ms=8, color=BLUE, mec=SURF, mew=2, zorder=3)
        elif r["label"] == "rated (beyond test)":
            ax.plot(r["t"], r["est"], "D", ms=8, mfc=SURF, mec=BLUE, mew=2, zorder=3)
        else:
            ax.plot(r["t"], r["est"], "X", ms=9, color=MUTED, zorder=3)
        if r["mc"] is not None:
            lo, hi = max(r["mc"] - 2 * r["mc_se"], r["mc"] * 1e-3), r["mc"] + 2 * r["mc_se"]
            ax.plot([r["t"], r["t"]], [lo, hi], color=INK, lw=1.2, zorder=4)
            ax.plot(r["t"], r["mc"], "_", ms=12, mew=2, color=INK, zorder=4)
    nun = sum(r["label"] == "unrated" for r in rs)
    if nun:
        msg = "all points unrated: three-way near-ties dominate" if nun == len(rs) else f"{nun} unrated point(s) (grey x)"
        ax.text(0.98, 0.96, msg, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(ylab, fontsize=9, color=INK2); ax.tick_params(colors=INK2, labelsize=8)


fig, axs = plt.subplots(2, 3, figsize=(13, 8.6), facecolor=SURF)
fig.suptitle("Safe-use rating, computed from the weights:\n"
             "if the 2nd and 3rd largest inputs differ by at least t, the chance of the error shown is about r(t)",
             fontsize=13, color=INK, x=0.01, ha="left", y=0.995)
tabs = []
for j, (nm, nice) in enumerate(MODELS):
    r1 = rows(nm, "E1", T1); r2 = rows(nm, "E2", T2); tabs.append((nm, nice, r1, r2))
    panel(axs[0, j], r1, f"{nice}\nP(answers with the 3rd largest | g23 >= t)", "t  (gap between 2nd and 3rd largest)")
    panel(axs[1, j], r2, f"{nice}\nP(any wrong answer | both top gaps >= t)\n(secondary; estimator not confirmed for this event)", "t  (g12 >= t and g23 >= t)")
axs[0, 0].set_ylabel("error chance under the condition", fontsize=9, color=INK2)
axs[1, 0].set_ylabel("error chance under the condition", fontsize=9, color=INK2)
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], color=BLUE, lw=2, marker="o", mec=SURF, mew=2, ms=8, label="rated, checked by conditional sampling"),
     Line2D([], [], color=BLUE, lw=0, marker="D", mfc=SURF, mec=BLUE, mew=2, ms=8, label="rated beyond test (too rare to sample)"),
     Line2D([], [], color=MUTED, lw=0, marker="X", ms=9, label="unrated (three-way near-tie dominates)"),
     Line2D([], [], color=INK, lw=1.2, marker="_", ms=12, mew=2, label="check: conditional sampling, +-2 s.e.")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(.5, 0.0))
C = {nm: E[nm]["C"] for nm, _ in MODELS}
fig.text(0.01, 0.035, "Cost of the rating (deductive, one run covers all t): " + ", ".join(f"{n} {C[k] / 1e6:.2f}M FPE" for k, n in
         (("orig", "M_16,10"), ("dev3_s529", "s529"), ("dev3_s531", "s531"))) +
         ".  Unverified; estimator = frozen H-R1 (K3, claude/t218-tie-tail-labels @e433919d).", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 0.93), h_pad=2.0)
fig.savefig(os.path.join(HERE, "results", "t401_rating.png"), dpi=130, facecolor=SURF)

L = ["# T401 safe-use rating table", "", "r(t) = estimated error chance under the condition; check = conditional-law MC rate (hits / n);",
     "plain MC needed = inputs plain sampling needs to see ~10 such errors.", ""]
for nm, nice, r1, r2 in tabs:
    for kind, rs, cond in (("E1", r1, "answers with the 3rd largest | g23 >= t"), ("E2", r2, "any wrong answer | g12, g23 >= t (unconfirmed)")):
        L += [f"## {nice}: P({cond})", "", "| t | P(condition) | rating r(t) | check (MC) | est / check | triple-tie share | label | plain MC needed |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            L.append(f"| {r['t']} | {r['P']:.3g} | {r['est']:.3g} | " + (f"{r['mc']:.3g} ({r['hits']} / {r['n']:.0e})" if r["mc"] is not None else f"< 30 hits in {r['n']:.0e}")
                     + f" | {r['ratio']:.2f} |" if r["ratio"] is not None else f"| {r['t']} | {r['P']:.3g} | {r['est']:.3g} | " + f"< 30 hits in {r['n']:.0e} | - |")
            L[-1] += f" {r['tie']:.2f} |" if r["tie"] is not None else " - |"
            L[-1] += f" {r['label']} | {r['plain_n10']:.2g} |"
        L.append("")
L.append("Cost (FPE, deductive): " + ", ".join(f"{k} {v:.3g}" for k, v in C.items()))
open(os.path.join(HERE, "results", "t401_rating.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
