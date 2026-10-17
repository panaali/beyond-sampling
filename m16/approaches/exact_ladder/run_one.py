"""python run_one.py T seed|name [check]  -> one JSON line with the exact accuracy, counts and FLOPs (appends to runs.jsonl)."""
import json
import sys

sys.path.append("../..")
import leaderboard as L  # noqa: E402
from ladder import exact_acc  # noqa: E402

T = int(sys.argv[1]); name = sys.argv[2]; check = len(sys.argv) > 3
ms = L.models(16, T)
m = ms[name]
a, info = exact_acc(m, check=check)
info.update(name=name, acc=a, fpe=info["flops"] / L.fpe_flops(16, T))
print(json.dumps(info), flush=True)
with open("runs.jsonl", "a") as f:
    f.write(json.dumps(info) + "\n")
