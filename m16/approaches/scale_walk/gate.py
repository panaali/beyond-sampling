"""Gate: est with compiled walk == est with frozen numpy walk (<= 1e-12 relative) at 4k FPE, first model of each
grid point (and every model given on the command line). -> gate.jsonl"""
import sys

import common as Cm

done = {r["tag"] for r in Cm.jl_read("gate.jsonl")}
first = {}
for r in Cm.registry():
    first.setdefault((r["H"], r["T"]), r["tag"])
tags = sys.argv[1:] or [t for t in first.values() if t not in done]
for tag in tags:
    m = Cm.load(tag)
    e0, c0 = Cm.frozen(compiled=False).est(m, 4000)
    e1, c1 = Cm.frozen(compiled=True).est(m, 4000)
    rel = max(abs(e1 - e0) / abs(e0), abs(c1 - c0) / abs(c0))
    row = {"tag": tag, "numpy": [e0, c0], "compiled": [e1, c1], "rel": rel, "pass": rel <= 1e-12}
    Cm.jl_add("gate.jsonl", row); print(row, flush=True)
