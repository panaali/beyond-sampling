"""Op attribution for fibre_fast.est_fast (exclusive counts). usage: python profile_fast.py H seed"""
import sys

import fibre
import fibre_fast as F
import profile_ops as PO
from fiber3 import Ops
from zoo import zoo_model


def wm(cls, name, label=None):
    f = getattr(cls, name)
    label = label or name

    def g(*a, **k):
        ops = next((x for x in list(a) + list(k.values()) if isinstance(x, Ops)), None)
        n0 = ops.n if ops is not None else 0
        PO.STACK.append([label, 0])
        try:
            return f(*a, **k)
        finally:
            nm, ch = PO.STACK.pop(); d = (ops.n - n0) if ops is not None else 0
            PO.TOT[nm] = PO.TOT.get(nm, 0) + d - ch; PO.CNT[nm] = PO.CNT.get(nm, 0) + 1
            if PO.STACK:
                PO.STACK[-1][1] += d
    setattr(cls, name, g)


F.rows_at = fibre.rows_at; F.trace_arc_rows = fibre.trace_arc_rows
for nm in ("forward_trace", "reverse_walk", "_clamp_piece", "_add_gate_cons", "_integrate"):
    PO.wrap(F, nm)
for nm in ("_flip", "use", "moved", "flipped", "finish"):
    wm(F.Piece, nm)


if __name__ == "__main__":
    H, s = int(sys.argv[1]), int(sys.argv[2])
    o = Ops(); inf = {}
    F.est_fast(zoo_model(H, 3, s), o, inf)
    print(f"H={H} seed {s}: total {o.n} traces {inf['traces']} cons/trace {inf['cons'] / inf['traces']:.1f}")
    for k in sorted(PO.TOT, key=lambda k: -PO.TOT[k]):
        print(f"  {k:16s} {PO.TOT[k]:9d} ({100 * PO.TOT[k] / o.n:4.1f}%) calls {PO.CNT[k]}")
