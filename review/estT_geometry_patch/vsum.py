"""T316: summarise val_k*_both_*.json (orig vs patched leaf sums, roots)."""
import json, sys
import numpy as np
for fn in sys.argv[1:]:
    d = json.load(open(fn)); o = np.array([v["orig"]["leaf"] for v in d.values()]); p = np.array([v["patched"]["leaf"] for v in d.values()])
    ro = np.array([v["orig"]["root"] for v in d.values()]); rp = np.array([v["patched"]["root"] for v in d.values()])
    st = {k: sum(v["patched"]["stats"][k] for v in d.values()) for k in ("cells_qhull", "cells_fallback", "cells_simplicial")}
    rel = np.abs(p - o) / np.maximum(o, 1e-300)
    print(fn, f"n={len(d)} sum orig {o.sum():.15f} patched {p.sum():.15f} | roots orig {ro.sum():.15f} patched {rp.sum():.15f}",
          f"| max|p-o| {np.abs(p - o).max():.2e} max rel {rel.max():.2e} (n rel>1e-9: {(rel > 1e-9).sum()}, of which mass>1e-6: {((rel > 1e-9) & (o > 1e-6)).sum()})",
          f"| max|leaf-root| orig {np.abs(o - ro).max():.2e} patched {np.abs(p - rp).max():.2e} | {st}")
