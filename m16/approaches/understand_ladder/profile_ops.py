"""Attribute the ops of a fibre tier to functions (exclusive counts). usage: python profile_ops.py H seed [tier]"""
import sys

import fibre
from fiber3 import Ops
from zoo import zoo_model

STACK = []
TOT = {}
CNT = {}


def wrap(mod, name):
    f = getattr(mod, name)

    def g(*a, **k):
        ops = next((x for x in list(a) + list(k.values()) if isinstance(x, Ops)), None)
        n0 = ops.n if ops is not None else 0
        STACK.append([name, 0])
        try:
            return f(*a, **k)
        finally:
            nm, child = STACK.pop()
            d = (ops.n - n0) if ops is not None else 0
            TOT[nm] = TOT.get(nm, 0) + d - child; CNT[nm] = CNT.get(nm, 0) + 1
            if STACK:
                STACK[-1][1] += d
    setattr(mod, name, g)


for nm in ("rows_at", "trace_arc_rows", "fibre_trace", "_walk", "trace_F2", "_integrate2"):
    wrap(fibre, nm)
_ci = fibre.Cons.interval


def ci(self, tm, t0, t1, ops):
    n0 = ops.n; r = _ci(self, tm, t0, t1, ops); d = ops.n - n0
    TOT["Cons.interval"] = TOT.get("Cons.interval", 0) + d; CNT["Cons.interval"] = CNT.get("Cons.interval", 0) + 1
    if STACK:
        STACK[-1][1] += d
    return r


fibre.Cons.interval = ci
_si = fibre.State.__init__


def si(self, model, x, P, U, nn, ops):
    n0 = ops.n; STACK.append(["State", 0])
    _si(self, model, x, P, U, nn, ops)
    nm, child = STACK.pop(); d = ops.n - n0
    TOT["State(excl rows_at)"] = TOT.get("State(excl rows_at)", 0) + d - child; CNT["State(excl rows_at)"] = CNT.get("State(excl rows_at)", 0) + 1
    if STACK:
        STACK[-1][1] += d


fibre.State.__init__ = si

if __name__ == "__main__":
    H, s = int(sys.argv[1]), int(sys.argv[2]); tier = sys.argv[3] if len(sys.argv) > 3 else "F2L2"
    m = zoo_model(H, 3, s); o = Ops()
    fibre.LOOK = 2 if tier == "F2L2" else 0
    fibre.est_fibre2(m, o)
    print(f"H={H} seed {s} {tier}: total {o.n}")
    for k in sorted(TOT, key=lambda k: -TOT[k]):
        print(f"  {k:22s} {TOT[k]:9d} ops ({100 * TOT[k] / o.n:4.1f}%)  calls {CNT[k]}")
    print(f"  {'(est_fibre2 itself)':22s} {o.n - sum(TOT.values()):9d}")
