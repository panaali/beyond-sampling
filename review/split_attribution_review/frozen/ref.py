"""Reference: unmodified estT (this folder's copy == estT_fix/estT.py) recipe108 -> ref/H{H}_T{T}_s{seed}.json.
usage: python ref.py H T seed"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import estT
from m43 import Ops
from zoo44 import zoo
H, T, seed = (int(v) for v in sys.argv[1:4])
o = Ops(); t0 = time.time(); acc = estT.estT(zoo(H, T, seed), o, mode="recipe108")
r = dict(acc=acc, ops=o.n, sec=time.time() - t0, **estT.STATS)
os.makedirs(os.path.join(HERE, "ref"), exist_ok=True)
json.dump(r, open(os.path.join(HERE, "ref", f"H{H}_T{T}_s{seed}.json"), "w"), indent=1); print(r)
