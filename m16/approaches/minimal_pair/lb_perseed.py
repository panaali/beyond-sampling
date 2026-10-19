"""Pre-registered pass check from the leaderboard raw runs: per-seed MSE ratio (single deterministic run per model)
against the leaderboard truths (4M) and, for T=5, against this folder's independent 48M truths."""
import json, sys
import numpy as np
T = int(sys.argv[1])
lt = json.load(open(f"../../../results/leaderboard_truths_H16_T{T}.json"))
own = {f"seed{k.split(':')[2]}": v for k, v in json.load(open("truths.json")).items() if k.startswith(f"16:{T}:")}
rows = [json.loads(l) for l in open("../../../data/leaderboard_v3_raw.jsonl")]
rows = {(r["model"], r["budget"]): r for r in rows if r["name"] == "minimal_pair" and r["T"] == T and r["H"] == 16}
for b in sorted({k[1] for k in rows}):
    out = []
    for s in range(5):
        r = rows[(f"seed{s}", b)]
        for tag, tr in (("lb", lt), ("own", own)):
            if f"seed{s}" not in tr: continue
            p, se = tr[f"seed{s}"]["acc"], tr[f"seed{s}"]["se"]
            out.append(f"s{s}/{tag} {((r['est']-p)**2 - se**2) / (p*(1-p)/r['cost']):+.3f}")
    print(f"T{T} budget {b}: per-seed MSE ratio  " + "  ".join(out))
