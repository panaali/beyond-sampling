"""Gate statistics and the results table from results_*.json and floor_*.json (no new computation)."""
import glob, json
import numpy as np

SCREEN = ["fused_s0", "fused_s1", "torch_s0", "torch_s1"]
res = {}
for f in sorted(glob.glob("results_*.json")):
    res.update(json.load(open(f)))
for d, k in [(2, 8), (3, 8), (4, 4)]:
    rows = [res.get(f"{nm}_d{d}k{k}") for nm in SCREEN]
    if None in rows:
        continue
    p = np.array([r["truth"] for r in rows]); s = np.array([r["s"] for r in rows])
    Cp = rows[0]["C_prop_fpe"]; Ce = rows[0]["C_end_fpe"]; C = max(Cp + Ce, 1000)
    pq = (p * (1 - p)).mean()
    g2 = ((np.abs(s - p) / 2) ** 2).mean() * 4000 / pq
    Fpt = ((s - p) ** 2).mean() * max(Cp, 1000) / pq
    print(f"d={d} k={k}: C_prop {Cp:.0f} FPE, C_end(2nd moments) {Ce:.0f}, total {Cp + Ce:.0f} | "
          f"s {np.round(s, 4).tolist()} truth {np.round(p, 4).tolist()} | RMS(s-p) {100 * np.sqrt(((s - p) ** 2).mean()):.1f} pp"
          f" | G2 stat {g2:.0f} (STOP if > 1) | point-F(s, C_prop) {Fpt:.0f} | Cantelli lower acc "
          f"{[r['cantelli_acc_lower'] for r in rows]} | rel L2 step10 {[round(r['rel_l2'][-1], 3) for r in rows]}")
print("\nD1 oracle floor: acc of argmax of the L2-optimal degree-d projection of the true logits (rel L2 of logits)")
for nm in SCREEN:
    fl = json.load(open(f"floor_{nm}.json"))
    print(nm, f"truth {fl[0]['truth']:.4f}", " ".join(f"d{r['d']}: {r['acc']:.3f} ({r['rel_l2_logits']:.2f})" for r in fl))
