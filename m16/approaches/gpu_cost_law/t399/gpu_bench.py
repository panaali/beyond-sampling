"""T399 GPU dry run: throughput and agreement of lpgpu's IPM engine (root bounds, no splitting) on the bench nodes.
(a) bench_182_k4 (273 nodes, k = 4) at DELTA = 1/3: root margin bound per node vs bab_cpu's exact-LP root margin
    (bab_d33_*.json, 26 nodes), prune count, and the 47 probe > 0 nodes, none of which may be pruned.
(b) local all-children benchmark (mkbench.py; T390 stages at DELTA = 0.1): prunes vs T390's LP; no event prefix pruned.
Timing per batch size; float64 IPM. usage: python gpu_bench.py BENCH.npz LOCAL.npz BAB_A.json,BAB_B.json [device]
-> gpu_bench.json"""
import json
import os
import sys
import time

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import lpgpu  # noqa: E402
from arc_eval import Wi0, Wh0, Wo0  # noqa: E402

dev = sys.argv[4] if len(sys.argv) > 4 else "cuda"; out = {}; LIM = int(os.environ.get("BENCH_N", 10 ** 9))  # smoke-test limit
E = lpgpu.LPEngine(Wi0, Wh0, Wo0, device=dev)


def run(G, A, k, tri, et):
    if dev == "cuda":
        th.cuda.synchronize()
    t0 = time.time(); mg, pr, lo, hi = E.prune(G, A, k, tri, event_tighten=et)
    if dev == "cuda":
        th.cuda.synchronize()
    return mg.cpu().numpy(), pr.cpu().numpy(), time.time() - t0


d = np.load(sys.argv[1]); tri = tuple(int(v) for v in d["tri"]); k = int(d["k"])
lpgpu.DELTA = 1 / 3
for et in (False, True):
    mg, pr, dt = run(d["G"][:LIM], d["A"][:LIM], k, tri, et)
    pos = d["probe_best"][:LIM] > 0
    ref = {r["idx"]: r["root_margin"] for f in sys.argv[3].split(",") for r in json.load(open(f))["nodes"]}
    diff = [(i, float(mg[i]), v) for i, v in ref.items() if i < len(mg)]
    out[f"bench_d33_et{int(et)}"] = dict(nodes=len(mg), sec=dt, nodes_per_s=len(mg) / dt, pruned=int(pr.sum()),
                                         probe_pos_pruned=int((pr & pos).sum()), vs_exact_root=diff)
    print("bench DELTA 1/3 event_tighten", et, "nodes %d sec %.1f pruned %d probe>0 pruned %d" % (len(mg), dt, pr.sum(), (pr & pos).sum()), flush=True)
    for i, a, b in diff:
        print("  node", i, "gpu %.5f exact %.5f" % (a, b), "UNSOUND" if (a < b - 1e-6 and not et) else "")
loc = np.load(sys.argv[2]); lpgpu.DELTA = 0.1
for kk in sorted(set(loc["k"].tolist())):
    sel = np.where(loc["k"] == kk)[0][:LIM]; ttri = tuple(int(v) for v in loc["tri"])
    for s0 in range(0, len(sel), 512):
        s = sel[s0:s0 + 512]
        mg, pr, dt = run(loc["G"][s], loc["A"][s], kk, ttri, False)
        t390 = loc["stage"][s] >= 0; ev = loc["is_event"][s]
        r = dict(nodes=len(s), sec=dt, nodes_per_s=len(s) / dt, t390_pruned=int(t390.sum()), gpu_pruned=int(pr.sum()),
                 both=int((pr & t390).sum()), gpu_only=int((pr & ~t390).sum()), t390_only=int((~pr & t390).sum()),
                 event_prefix_pruned=int((pr & ev).sum()), event_prefixes=int(ev.sum()))
        out[f"local_k{kk}_{s0}"] = r; print("local k", kk, r, flush=True)
json.dump(out, open(os.path.join(HERE, "gpu_bench.json"), "w"), indent=1)
