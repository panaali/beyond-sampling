"""T392 summary: table (markdown, stdout) and plot of F against k (results/F_vs_k.png).

usage: python summarize.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = [(1, 24), (2, 48), (5, 128), (10, 128)]  # k1h32 (post-hoc) is in the README only


def main():
    rows, pts = [], {}
    for k, H in CONF:
        p = f"{HERE}/results/score_k{k}h{H}.json"
        if not os.path.exists(p) or "summary_random" not in json.load(open(p)):
            continue
        r = json.load(open(p)); s = r["summary_random"]; h = r.get("halton", {})
        pat = [v[-1]["distinct"] for v in r["patterns"].values()]
        f = lambda x: f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"  # noqa: E731
        hal = {C: h.get(f"generic_C{C}_family", {}).get("F_sets") for C in (4000, 16000)}
        hs = lambda v: "/".join(f"{x:.2g}" for x in v) if v else "-"  # noqa: E731
        rows.append(f"| {k} | {H} | {r['fpe']} | {s['acc_mean']:.4f} | {min(pat)/1e6:.2f}-{max(pat)/1e6:.2f}M | {s['pieces']:.0f} | "
                    f"{s['cost_fpe']:.0f} | {f(s['F'])} | {f(s['F_direct_4000'])} | {f(s['F_direct_16000'])} | {f(s['F_incr'])} | "
                    f"{s['F_floor'][0]:.3f} | {hs(hal[4000])} | {hs(hal[16000])} |")
        pts[k] = (s, h)
        if k == 10:
            fh = {C: h.get(f"fast1_C{C}_family", {}).get("F_sets") for C in (4000, 16000)}
            rows.append(f"| 10 (fast1) | {H} | | | | | {r['random']['k10h128_s0']['fast1']['cost_fpe']:.1f} (s0) | {f(s['F_fast1'])} | | | | | "
                        f"{hs(fh[4000])} | {hs(fh[16000])} |")
    print("| k | H | FPE | panel acc | gate patterns (1e7, 3 seeds) | pieces/circle | cost/circle (FPE) | F random | F direct 4k | "
          "F direct 16k | F_incr | F_floor | Halton F 4k (5 sets) | Halton F 16k (5 sets) |")
    print("|" + "---|" * 14)
    print("\n".join(rows))
    ks = sorted(pts)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for key, lab, mk in (("F", "random circles, generic walk (implemented)", "o"), ("F_incr", "random circles, incremental cost model", "s"),
                         ("F_floor", "random circles, floor cost", "^")):
        y = [pts[k][0][key][0] for k in ks]
        lo = [y[i] - pts[k][0][key][1] for i, k in enumerate(ks)]; hi = [pts[k][0][key][2] - y[i] for i, k in enumerate(ks)]
        ax.errorbar(ks, y, yerr=[lo, hi], marker=mk, capsize=3, label=lab)
    for C, mk in ((4000, "x"), (16000, "+")):
        for k in ks:
            v = pts[k][1].get(f"generic_C{C}_family", {}).get("F_sets", [])
            ax.scatter([k] * len(v), v, marker=mk, color="grey", alpha=.6, label=f"Halton sets, generic, C={C//1000}k" if k == ks[0] else None)
    if 10 in pts:
        s = pts[10][0]["F_fast1"]
        ax.errorbar([10], [s[0]], yerr=[[s[0] - s[1]], [s[2] - s[0]]], marker="D", color="C3", capsize=3, label="k=10 fast1 (implemented incremental)")
        for C, mk in ((4000, "1"), (16000, "2")):
            v = pts[10][1].get(f"fast1_C{C}_family", {}).get("F_sets", [])
            ax.scatter([10] * len(v), v, marker=mk, color="C3", label=f"k=10 fast1 Halton, C={C//1000}k")
    ax.axhline(1, color="k", lw=.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (inputs per step; 10/k steps)"); ax.set_ylabel("F (MSE x cost / p(1-p); sampling = 1)")
    ax.set_title("T392: walk F vs plain sampling, simplest >= .95 config per k", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1)); fig.tight_layout(); fig.savefig(f"{HERE}/results/F_vs_k.png", dpi=130)


if __name__ == "__main__":
    main()
