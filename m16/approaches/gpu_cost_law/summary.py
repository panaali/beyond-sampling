"""T192/T193 tables from results/ -> results/summary.json and printed markdown.

Main table: halves-start exact (E46/E48-comparable) vs recipe108. err = recipe108 - exact.
Cost columns (T196 review): estT charged ops; charged x 1.37 (nominal audit at (8,4), transferred); own nominal audit
FLOPs (audit.py) where measured. MSE ratio = err^2 / (p (1 - p) / N), N = cost / cost of one sample:
  ops column  : N = charged ops / leaderboard.fpe_flops (E46 convention)
  FLOPs column: N = (audited FLOPs, else 1.37 x charged) / T196 per-sample FLOPs (682 at (8,4), 853 at (8,5))
  wall column : N = recipe wall seconds x float64 MC samples/s of mc_gpu.py on the same A100
Second table: exact variants (truth-cone start, sliver bisection) with leaf sums. Third: noisy-copy cost.
Fourth: M_16,10 prefix regions. Fifth: cost law (cells, regions, ops vs T)."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from leaderboard import fpe_flops  # noqa: E402

SAMPLE_FLOPS = {(8, 4): 682, (8, 5): 853}
AUD_FACTOR = 1.37


def variant(r):
    return ("truth" if r.get("start") == "truth" else "halves") + ("+sliver" if r.get("sliver") else "")


R = {}
for f in glob.glob(os.path.join(HERE, "results", "runs", "*.json")):
    r = json.load(open(f))
    if r.get("halves_run", r.get("halves")) != r.get("halves") or r.get("acc") is None:
        continue
    R[(r["H"], r["T"], str(r["model"]).replace(":", "_"), r["mode"], variant(r), bool(r.get("leafsum_minus_1") is not None))] = r


def get(H, T, m, mode, var="halves"):
    return R.get((H, T, m, mode, var, False)) or R.get((H, T, m, mode, var, True))


AUD = {}
for f in glob.glob(os.path.join(HERE, "results", "audit", "*.json")):
    r = json.load(open(f)); AUD[(r["H"], r["T"], r["model"].replace(":", "_"), r["mode"])] = r
MC = {}
for sub in ("mc", "mc2"):
    for f in glob.glob(os.path.join(HERE, "results", sub, "*.json")):
        r = json.load(open(f)); MC.setdefault((r["H"], r["T"], r["model"].replace(":", "_")), []).append(r)

f_ = lambda v, s: (s % v) if v is not None else "-"
out = dict(main=[], variants=[], noisy=[], prefix=[], costlaw=[])
print("### Main: halves-start exact vs recipe108\n")
print("| (H,T) | model | exact acc | MC z (each stream) | recipe err pp | ops exact / recipe (charged) | recipe x1.37 / audited FLOPs | cells exact / recipe | regions | MSE ratio ops / FLOPs / wall |")
print("|---|---|---|---|---|---|---|---|---|---|")
models = sorted(set((k[0], k[1], k[2]) for k in R if k[3] == "exact" and k[4] == "halves"))
for H, T, m in models:
    ex = get(H, T, m, "exact"); rec = get(H, T, m, "recipe108"); p = ex["acc"]
    zs = [(mc["acc"] - p) / mc["se"] for mc in MC.get((H, T, m), [])]
    row = dict(H=H, T=T, model=m, exact=p, exact_ops=ex["ops"], exact_cells=ex["cells"], regions=ex.get("regions"),
               z_mc=zs, exact_wall=ex["sec_wall"])
    if rec:
        err = rec["acc"] - p; a = AUD.get((H, T, m, "recipe108"))
        row.update(recipe=rec["acc"], err_pp=100 * err, recipe_ops=rec["ops"], recipe_cells=rec["cells"],
                   recipe_ops_x137=AUD_FACTOR * rec["ops"], recipe_flops_audit=a["flops_complete"] if a else None,
                   mse_ratio_ops=err ** 2 / (p * (1 - p) / (rec["ops"] / fpe_flops(H, T))))
        if (H, T) in SAMPLE_FLOPS:
            fl = a["flops_complete"] if a else AUD_FACTOR * rec["ops"]
            row.update(mse_ratio_flops=err ** 2 / (p * (1 - p) / (fl / SAMPLE_FLOPS[(H, T)])), flops_audited=bool(a))
        mcs = MC.get((H, T, m))
        if mcs:
            rate = mcs[0]["n"] / mcs[0]["sec"]
            row["mse_ratio_wall"] = err ** 2 / (p * (1 - p) / (rec["sec_wall"] * rate))
    out["main"].append(row)
    print(f"| ({H},{T}) | {m} | {p:.10f} | {' '.join('%+.2f' % z for z in zs) or '-'} | {f_(row.get('err_pp'), '%+.6f')} | "
          f"{row['exact_ops']:.3g} / {f_(row.get('recipe_ops'), '%.3g')} | {f_(row.get('recipe_ops_x137'), '%.3g')} / "
          f"{f_(row.get('recipe_flops_audit'), '%.3g')} | {row['exact_cells']:.0f} / {f_(row.get('recipe_cells'), '%.0f')} | "
          f"{f_(row.get('regions'), '%d')} | {f_(row.get('mse_ratio_ops'), '%.2g')} / "
          f"{f_(row.get('mse_ratio_flops'), '%.2g')}{'' if row.get('flops_audited', True) else '*'} / "
          f"{f_(row.get('mse_ratio_wall'), '%.2g')} |")

print("\n### Exact variants (truth-cone start, sliver bisection) and leaf sums\n")
print("| (H,T) | model | variant | acc | leaf sum - 1 | cells | regions | ops | geometry ops | simplices (thin) | wall s |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(R):
    H, T, m, mode, var, ls = k; r = R[k]
    if mode != "exact" or ("rel" in m or "abs" in m) or (var == "halves" and not ls):
        continue
    out["variants"].append(dict(H=H, T=T, model=m, variant=var, acc=r["acc"], leaf=r.get("leafsum_minus_1"),
                                cells=r["cells"], regions=r.get("regions"), ops=r["ops"]))
    print(f"| ({H},{T}) | {m} | {var} | {r['acc']:.10f} | {f_(r.get('leafsum_minus_1'), '%.1e')} | {r['cells']:.0f} | "
          f"{f_(r.get('regions'), '%d')} | {r['ops']:.3g} | {f_(r.get('ops_geometry'), '%.3g')} | "
          f"{r['simplices']:.3g} ({f_(r.get('thin'), '%.3g')}) | {r['sec_wall']:.0f} |")

print("\n### Noisy copies: exact (truth start) cost relative to the clean model\n")
print("| (H,T) | copy | acc | cells | regions | ops | cells / regions / ops vs clean |")
print("|---|---|---|---|---|---|---|")
for H in (8, 16):
    c = R.get((H, 4, "0", "exact", "truth", True))
    if not c:
        continue
    for k in sorted(R):
        if k[0] == H and k[1] == 4 and ("rel" in k[2] or "abs" in k[2] or k[2] == "0") and k[4] == "truth" and k[5]:
            r = R[k]
            out["noisy"].append(dict(H=H, model=k[2], acc=r["acc"], cells=r["cells"], regions=r["regions"], ops=r["ops"]))
            print(f"| ({H},4) | {k[2]} | {r['acc']:.4f} | {r['cells']:.0f} | {r['regions']} | {r['ops']:.3g} | "
                  f"{r['cells'] / c['cells']:.2f} / {r['regions'] / c['regions']:.2f} / {r['ops'] / c['ops']:.2f} |")

print("\n### M_16,10 prefixes (first k recurrent steps): exact regions (prefix.py)\n")
print("| k | regions | cells | growth | geometry ops | CPU s | complete |")
print("|---|---|---|---|---|---|---|")
prev = None
for f in sorted(glob.glob(os.path.join(HERE, "results", "prefix", "k*.json")), key=lambda s: int(s.split("k")[-1][:-5])):
    r = json.load(open(f)); out["prefix"].append(r)
    reg = r["regions"]; cells = r.get("cells_est", r["cells"])
    print(f"| {r['k']} | {reg}{'' if r['complete'] else ' (lower bound)'} | {cells:.4g}{'' if r['complete'] else ' (est.)'} | "
          f"{f_(reg / prev if prev and r['complete'] else None, 'x%.1f')} | {r['ops_geometry']:.3g} | {r['sec_cpu']:.0f} | {r['complete']} |")
    prev = reg

print("\n### Cost law along T (exact, halves start unless noted; mean over seeds with min..max)\n")
print("| (H,T) | seeds | cells | regions | ops | simplices | cells/regions |")
print("|---|---|---|---|---|---|---|")
for H, T in sorted(set((k[0], k[1]) for k in R)):
    rows = [R[k] for k in R if k[0] == H and k[1] == T and k[3] == "exact" and k[4] in ("halves", "truth")
            and k[2].isdigit()]
    for var in ("halves", "truth"):
        rr = [r for r in rows if variant(r).startswith(var)]
        rr = {str(r["model"]): r for r in rr}.values()
        rr = list(rr)
        if not rr:
            continue
        g = lambda key: [r.get(key) for r in rr if r.get(key) is not None]
        fm = lambda v: f"{sum(v) / len(v):.3g} ({min(v):.3g}..{max(v):.3g})" if v else "-"
        cr = [r["cells"] / r["regions"] for r in rr if r.get("regions")]
        out["costlaw"].append(dict(H=H, T=T, start=var, n=len(rr), cells=g("cells"), regions=g("regions"), ops=g("ops")))
        print(f"| ({H},{T}) {var} | {len(rr)} | {fm(g('cells'))} | {fm(g('regions'))} | {fm(g('ops'))} | "
              f"{fm(g('simplices'))} | {fm(cr)} |")
json.dump(out, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1, default=float)
