"""T426: pre-registered triple choice from the MC summaries. usage: python pick.py TAG T0 -> 'A B C' (A = '-' if no hits)"""
import glob
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
tag, t0 = sys.argv[1], float(sys.argv[2])
C = Counter(); T = None
for f in glob.glob(os.path.join(HERE, "results", f"mc_{tag}_{t0:g}_*.json")):
    d = json.load(open(f)); T = d["T"]
    for t, c in d["by_triple"]:
        C[tuple(t)] += c
rank = sorted(C.items(), key=lambda r: (-r[1], sorted(r[0])[1]))      # most hits; tie: smaller b
fmt = lambda t: ",".join(map(str, t))
A = rank[0][0] if rank else None; B = (1, 8, 2)
others = [r for r in rank if r[0] not in (A, B)]
Cc = others[0][0] if others and others[0][1] >= 2 else (T - 1, T - 3, T - 2)
print(fmt(A) if A else "-", fmt(B), fmt(Cc))
