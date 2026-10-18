"""T205: restate published cells/regions ratios against exact (certified) region counts. Pure arithmetic on saved files:
regions/*.json (regions_exact.py), recount112.json, mc.json (lb), res/*.json, estT_fix/rescore + run5, understand_m43
evalH_*.json, final3/summary.json. Means over the 5 clean seeds (zoo0 only at (8,5)). -> restate.json"""
import json, os, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); A = os.path.join(HERE, "..")
J = lambda *p: json.load(open(os.path.join(*p)))
reg = {os.path.basename(f)[:-5]: J(f) for f in glob.glob(os.path.join(HERE, "regions", "*.json"))}
mc = J(HERE, "mc.json"); r112 = J(HERE, "recount112.json"); f3 = J(A, "final3", "summary.json")
mean = lambda xs: float(np.mean(xs))
rows = []


def add(label, H, T, seeds, pub_cells, pub_regions, cells, published=None):
    k = [f"H{H}_T{T}_zoo{s}" for s in seeds]
    ex = mean([reg[q]["regions_certified"] for q in k]); lb = mean([mc[q]["regions_lb_2e5"] for q in k])
    rows.append(dict(label=label, H=H, T=T, seeds=len(seeds), published_ratio=published or pub_cells / pub_regions,
                     published_cells=pub_cells, published_regions=pub_regions, cells=cells, regions_exact=ex,
                     regions_lb2e5=lb, lb_undercount=ex / lb, restated_ratio=cells / ex,
                     violations=sum(reg[q]["cert_violations"] for q in k), center_equal=all(reg[q]["regions_center"] == reg[q]["regions_certified"] for q in k)))


S = range(5)
for H in (4, 5, 6, 8):                                     # H-U4 / E43: RECIPE112 clean cells (doubled) / exact3 regions
    ev = J(A, "understand_m43", f"evalH_{H}.json")["zoo"]
    add("E43 H-U4 RECIPE112", H, 3, S, mean([ev[str(s)]["clean_cells"] for s in S]), mean([ev[str(s)]["regions"] for s in S]),
        mean([r112[f"H{H}_T3_zoo{s}"]["cells"] for s in S]))
add("RECIPE112 (this folder)", 16, 3, S, None, None, mean([r112[f"H16_T3_zoo{s}"]["cells"] for s in S]), published=float("nan"))
z3 = f3["T3"]                                               # E45 T trend: T3 used E43 cells and lb regions
add("E45 T-trend", 8, 3, S, mean([z3[f"zoo{s}"]["cells"] for s in S]), mean([z3[f"zoo{s}"]["regions_lb"] for s in S]),
    mean([J(HERE, "res", f"H8_T3_zoo{s}_recipe108.json")["cells"] for s in S]))
rc4 = [J(A, "estT_fix", "rescore", f"zoo{s}.json")["recipe108"]["cells"] for s in S]
add("E45/E46 recipe108", 8, 4, S, mean(rc4), mean([f3["T4"][f"zoo{s}"]["regions_lb"] for s in S]), mean(rc4))
if "H8_T5_zoo0" in reg:
    c5 = J(A, "estT_fix", "run5", "zoo0_recipe108.json")["cells"]
    add("E48 recipe108", 8, 5, [0], c5, f3["T5"]["H8_T5_zoo0_recipe108"]["regions_lb"], c5)
for T in (3, 4):
    c = mean([J(HERE, "res", f"H16_T{T}_zoo{s}_recipe108.json")["cells"] for s in S])
    add("T194 recipe108", 16, T, S, None, None, c, published=float("nan"))
print("| source | (H,T) | published cells / regions = ratio | cells (corrected) | exact regions | lb2e5 (undercount) | restated ratio |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    pub = "—" if r["published_cells"] is None else f"{r['published_cells']:.0f} / {r['published_regions']:.0f} = {r['published_ratio']:.2f}"
    print(f"| {r['label']} | ({r['H']},{r['T']}) | {pub} | {r['cells']:.0f} | {r['regions_exact']:.0f} | {r['regions_lb2e5']:.0f} (x{r['lb_undercount']:.2f}) | **{r['restated_ratio']:.2f}** |")
print("violations", sum(r["violations"] for r in rows), "center==certified", all(r["center_equal"] for r in rows))
json.dump(rows, open(os.path.join(HERE, "restate.json"), "w"), indent=1)
