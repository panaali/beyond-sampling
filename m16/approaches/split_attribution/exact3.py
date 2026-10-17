"""Exact accuracy (estT mode 'exact', fixed masses) at (8,3) seeds 0-4 -> ref/exact_H8_T3.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro")]
import estT
from m43 import Ops
from zoo44 import zoo
r = {s: estT.estT(zoo(8, 3, s), Ops(), mode="exact") for s in range(5)}
json.dump(r, open(os.path.join(HERE, "ref", "exact_H8_T3.json"), "w"), indent=1); print(r)
