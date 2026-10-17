"""After the T42 audit fixes (tie rule, AUGMENT cell degrees, full symmetrisation): re-run the configurations that
can change on the base models and print old-table vs new-table estimates. Usage: python recheck_fixes.py T"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "m16")
import arc_kprop as A, leaderboard as LB
from mlp_kprop import kprop_harmonic as KH
T = int(sys.argv[1]); name, m = LB._base(16, T)
print(name, "uncond K2 (tie rule change only):", A.estimate_uncond(m, 2)[0])
for kind, K in [("SIMPLE", 2), ("AUGMENT", 2)] + ([("SIMPLE", 3)] if T < 10 else []):
    o = min(4, KH.get_d_max(K, getattr(KH, kind)))
    old = A.estimate_cell(m, K, kind, base=A.cell_base_cumulants(T, order=o, full_sym=False))[0]
    new = A.estimate_cell(m, K, kind, base=A.cell_base_cumulants(T, order=o, full_sym=True))[0]
    print(f"cell {kind} K={K}: dihedral-sym table {old:.5f}  full-sym table {new:.5f}", flush=True)
