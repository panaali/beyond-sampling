"""As-implemented work vs charged cost for the segment-walk estimators (orig model, budget 16000).
The walkers charge a linear cell 2(T-t)/T FPE (value+tangent recomputed only from the flipped step t), but the
code recomputes all T steps (2 FPE per cell). This script counts rows passed through the full value+tangent
recurrence and reports implemented_FPE / charged_FPE. usage (repo root): python review/estimator_audit/fullpass_count.py"""
import sys, importlib, json
import numpy as np
sys.path.insert(0, "m16"); sys.path.insert(0, "m16/approaches/tube")
from arc_eval import Wi0, Wh0, Wo0
M = (Wi0, Wh0, Wo0)
out = {}

# tube and sci_laplace share tube.fwd_jvp / tube.forward_from
import tube as TB
cnt = dict(jvp=0, fwd=0)
_j, _f = TB.fwd_jvp, TB.forward_from
def jvp(model, x, dx): cnt["jvp"] += len(x); return _j(model, x, dx)
def fwd(model, x): cnt["fwd"] += len(x); return _f(model, x)
TB.fwd_jvp = jvp; TB.forward_from = fwd
spec = importlib.util.spec_from_file_location("e_tube", "m16/approaches/tube/estimator.py"); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
e.forward_from = fwd
_, c = e.est(M, 16000)
out["tube"] = dict(charged=c, implemented=2 * cnt["jvp"] + cnt["fwd"], ratio=(2 * cnt["jvp"] + cnt["fwd"]) / c)

sys.path.insert(0, "m16/approaches/sci_laplace")
import lqt
lqt.forward_from = fwd
import tube as TB2  # same module object
cnt.update(jvp=0, fwd=0)
spec = importlib.util.spec_from_file_location("e_lap", "m16/approaches/sci_laplace/estimator.py"); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
_, c = e.est(M, 16000)
out["sci_laplace"] = dict(charged=c, implemented=2 * cnt["jvp"] + cnt["fwd"], ratio=(2 * cnt["jvp"] + cnt["fwd"]) / c)

# sci_tao: cells returned by walk_mass
for k in [k for k in list(sys.modules) if k in ("core",)]: del sys.modules[k]
sys.path.insert(0, "m16/approaches/sci_tao")
import core as CT
tot = dict(cells=0)
_w = CT.walk_mass
def wm(*a, **k):
    F, c, cells = _w(*a, **k); tot["cells"] += int(cells.sum()); return F, c, cells
CT.walk_mass = wm
spec = importlib.util.spec_from_file_location("e_tao", "m16/approaches/sci_tao/estimator.py"); e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
_, c = e.est(M, 16000)
out["sci_tao"] = dict(charged=c, implemented=2 * tot["cells"], ratio=2 * tot["cells"] / c)
print(json.dumps(out, indent=1))
