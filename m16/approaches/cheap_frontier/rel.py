"""reliability (m16/approaches/reliability/estimator.py) at its LB3 ladder rung L0 and cheaper variants of L0.
Cost as the reliability estimator counts it (C.fwd x 1.3 Rosenblatt overhead + SORM FD Hessians), in its FPE.
Labels: design points by deterministic search on exact gradients, FORM/SORM, Ditlevsen bounds = deductive
approximations; chamber subset (fixed arrangement of ranks 4-10, stride over triples) = INDUCTIVE."""
import os
import sys

_rel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reliability")
sys.path.insert(0, _rel)
for _m in ("form", "rosen", "sorm", "system"):
    sys.modules.pop(_m, None)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("rel_estimator", os.path.join(_rel, "estimator.py"))
RE = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RE)

L0 = RE.LADDER[0]
VARIANTS = {
    "L0": {},
    "L0_s2": dict(stride_pos=8, stride_neg=16),
    "L0_s4": dict(stride_pos=16, stride_neg=32),
    "L0_s8": dict(stride_pos=32, stride_neg=64),
    "L0_m02": dict(modes_pos=(0, 2), modes_neg=(0, 2)),
    "L0_nosorm": dict(sorm=False),
    "L0_s4_m02_nosorm": dict(stride_pos=16, stride_neg=32, modes_pos=(0, 2), modes_neg=(0, 2), sorm=False),
}


def est_rel(model, variant="L0", return_parts=False):
    acc, cost, parts = RE.estimate(model, {**L0, **VARIANTS[variant]})
    return (acc, cost, parts) if return_parts else (acc, cost)
