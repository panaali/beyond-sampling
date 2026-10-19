"""Two-tier rigorous charging of a survivor slice (pre-registration 3). Tier (i): M2 on every cell. Tier (ii):
greedily move the cell with the largest current ub to the next M5t budget (200 -> 2000 -> 8000 simplices),
ub = min(old, new); stop when the top cell is < 1% of Σub or the CPU budget is spent. Every stage is rigorous.
Works on a shard [i0, i1) of the survivor file; shards add.
usage: python tiered.py SURV_FILE t0 cpu_hours i0 i1 -> results_tiered_<name>_<t0>_<i0>.json"""
import heapq
import json
import os
import sys
import time

import numpy as np

import rigour as R

f, t0, hours, i0, i1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
c = np.load(f, allow_pickle=True)
tri = tuple(int(v) for v in c["tri"]); k = int(c["k"]); norm = 1 / R.q(float(c["delta"]))
G = c["gates"][i0:i1].astype(int)
BUDGETS = [200, 2000, 8000]
T0 = time.process_time()
cells = []
for j, g in enumerate(G):
    t = time.process_time()
    try:
        ub = R.mass_upper(tri, g, k, t0, norm=norm)
    except Exception:
        ub = 1.0                                      # trivially valid
    cells.append(dict(i=i0 + j, ub=ub, lb=0.0, tier=0, t=time.process_time() - t))
hist = [dict(stage="M2", sum_ub=sum(x["ub"] for x in cells), cpu=time.process_time() - T0)]
print(hist[-1], flush=True)
heap = [(-x["ub"], j) for j, x in enumerate(cells)]; heapq.heapify(heap)
refined = {b: 0 for b in BUDGETS}
while heap and time.process_time() - T0 < hours * 3600:
    tot = sum(x["ub"] for x in cells)
    negub, j = heapq.heappop(heap)
    x = cells[j]
    if -negub < 0.01 * tot:
        break
    if x["tier"] >= len(BUDGETS):
        continue                                          # fully refined: leave it out of the heap
    b = BUDGETS[x["tier"]]
    t = time.process_time()
    try:
        lb, ub = R.mass_cell(tri, G[j], k, t0, budget=b, tangent=True, zbins=24, zratio=R.fmpq(23, 20), norm=norm)
        x["ub"] = min(x["ub"], ub); x["lb"] = max(x["lb"], lb)
    except Exception as e:
        x["err"] = repr(e)
    x["tier"] += 1; x["t"] += time.process_time() - t; refined[b] += 1
    heapq.heappush(heap, (-x["ub"], j))
    hist.append(dict(stage=f"M5t{b}", cell=x["i"], sum_ub=sum(y["ub"] for y in cells), cpu=time.process_time() - T0))
    if len(hist) % 20 == 0:
        print(hist[-1], flush=True)
out = dict(file=os.path.basename(f), t0=t0, i0=i0, i1=i1, cells=cells, hist=hist, refined=refined,
           sum_ub=sum(x["ub"] for x in cells), sum_lb=sum(x["lb"] for x in cells), cpu=time.process_time() - T0)
json.dump(out, open(f"results_tiered_{os.path.basename(f)[:-4]}_{t0}_{i0}.json", "w"), indent=0)
print("DONE sum_ub %.4g (M2 %.4g) sum_lb %.4g refined %s cpu_h %.2f" % (out["sum_ub"], hist[0]["sum_ub"], out["sum_lb"], refined, out["cpu"] / 3600))
