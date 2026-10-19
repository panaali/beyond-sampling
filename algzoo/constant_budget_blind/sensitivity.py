"""Pre-declared sensitivity: per-size results excluding the flagged run wfpn (README protocol log)."""
import json
from pathlib import Path
import numpy as np
import analyze as A  # reuses parsed judge outputs

EX = "wfpn"
out = {}
for H in (8,):
    codes = [c for c in A.audit if A.audit[c]["H"] == H and c != EX]
    G = float(np.median([A.score[c]["gap_recovered"] for c in codes]))
    m = [A.pair_score(it) for it in A.pair_items if it["H"] == H and EX not in it["pair"] and it["kind"] == "matched"]
    k = [A.pair_score(it) for it in A.pair_items if it["H"] == H and EX not in it["pair"] and it["kind"] == "control"]
    out[H] = {"G": G, "S_matched": float(np.mean(m)), "S_control": float(np.mean(k)), "D": float(np.mean(m) - np.mean(k)), "n_pairs": len(m)}
s = A.summary["sizes"]
G32, D32, lo = s[32]["G"], s[32]["D"], s[32]["D_CI95"][0]
G8, D8 = out[8]["G"], out[8]["D"]
rec = "holds" if G32 >= .9 and G32 >= G8 - .05 else "degrades" if G32 < .9 and G32 < G8 - .05 else "mixed"
lab = "holds" if D32 >= D8 - .15 and lo > 0 else "degrades" if D32 < D8 - .15 else "mixed"
out["verdict_excluding_wfpn"] = {"reconstruction": rec, "labels": lab}
json.dump(out, open(Path(__file__).parent / "results" / "sensitivity_excl_wfpn.json", "w"), indent=1)
print(json.dumps(out, indent=1))
