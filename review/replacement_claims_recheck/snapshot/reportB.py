"""Print Phase B tables from results/phaseB.json."""
import json
import numpy as np

R = json.load(open("results/phaseB.json"))
F = R["families"]
r2 = np.array([[R["r2"][str(t)][f] for f in F] for t in range(1, 11)])   # (t, fam, j)
short = [f.split(":")[0] for f in F]
print("Families:", "; ".join(F), "\n")
print("**Held-out R^2 of pre_t[j], best family; and the first (simplest) family reaching R^2 >= 0.99**\n")
print("| neuron | " + " | ".join(f"t={t}" for t in range(1, 11)) + " |")
print("|---" * 11 + "|")
for j in range(16):
    cells = []
    for ti in range(10):
        v = r2[ti, :, j]
        ok = np.where(v >= 0.99)[0]
        cells.append(f"{v.max():.3f} ({short[ok[0]] if len(ok) else '-'})")
    print(f"| n{j} | " + " | ".join(cells) + " |")
print("\n**Held-out R^2 at t=10 by family**\n")
print("| neuron | " + " | ".join(short) + " |")
print("|---" * (len(F) + 1) + "|")
for j in range(16):
    print(f"| n{j} | " + " | ".join(f"{r2[9, f, j]:.3f}" for f in range(len(F))) + " |")
print("| median | " + " | ".join(f"{np.median(r2[9, f]):.3f}" for f in range(len(F))) + " |")
print("\n**Pooled vs within-pattern fits at t=10 (pattern = gates of h_9); median over neurons active in pattern**\n")
for g, rec in R["pattern"].items():
    act = [j for j in range(16) if int(g) >> j & 1]
    line = [f"pattern {g} ({100 * rec['share']:.1f}%)"]
    for k, v in rec.items():
        if k == "share":
            continue
        if "within_pattern" in v:
            w = np.median(np.array(v["within_pattern"])[act])
            pm = np.median(np.array(v["pooled_model"])[act]) if "pooled_model" in v else None
            line.append(f"{k.split(':')[0]} within {w:.3f}" + (f" / pooled {pm:.3f}" if pm is not None else ""))
    print("- " + "; ".join(line))
M = R["modes64975"]
print("\n**Mode coordinates of h_10 in final pattern 64975 (held-out R^2)**\n")
keys = [k for k in M if k not in ("share", "coords")]
print("| coordinate | " + " | ".join(k.split(":")[0] for k in keys) + " |")
print("|---" * (len(keys) + 1) + "|")
for i, c in enumerate(M["coords"]):
    print(f"| {c} | " + " | ".join(f"{M[k][i]:.3f}" for k in keys) + " |")
