"""Parity of nonzero K4/feedback slices and ragged row blocks, not scores."""
import importlib.util
import json
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import pytest

from estimator_blocked import Estimator as Blocked

HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "sources.json").read_text())["v25"]["local_path"]
spec = importlib.util.spec_from_file_location("original_v25_control", source)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


@pytest.mark.parametrize("n,ka,kb,rfb,need_d21", [
    (129, 0, 0, 2, True), (257, 2, 0, 2, True),
    (257, 2, 1, 2, True), (129, 3, 3, 2, True),
    (129, 2, 1, 0, True), (257, 2, 1, 2, False),
])
def test_ragged_rows_feedback_and_k4(n, ka, kb, rfb, need_d21):
    rng = np.random.default_rng(474+n+ka+kb+rfb)
    k, rres, r1, r2 = 3, 3, 5, 2

    def arr(*shape):
        return fnp.asarray((rng.normal(size=shape)*0.05).astype(np.float32))

    A, P = arr(k,n,n), arr(k,n,n)
    Z, L = arr(k,n,rres+2), arr(k,n,rres+2)
    lists = [[arr(n) for _ in range(k)] for _ in range(6)]
    Zf = arr(k,n,2*rfb) if rfb else None
    R1 = arr(k,n,rfb) if rfb else None
    R2 = arr(k,n,rfb) if rfb else None
    extra = dict(need_d21=need_d21, ka=ka, kb=kb,
                 FAo=arr(ka-kb,r1,n) if ka>kb else None,
                 FPo=arr(ka-kb,r1,n) if ka>kb else None,
                 Qc=arr(n,r1) if ka else None,
                 FA2=arr(kb,r2,n) if kb else None,
                 FP2=arr(kb,r2,n) if kb else None,
                 U2=arr(r1,r2) if kb else None)
    outputs = []
    for cls, rows in [(original.Estimator, n), (Blocked, min(128,n))]:
        scratch = {name: fnp.empty((k, rows, n), dtype=fnp.float32)
                   for name in ("ap", "pp", "t", "mp", "la", "lp", "xt", "yt", "u")}
        with flops.BudgetContext(flop_budget=10**12, quiet=True):
            out = cls()._dslices(A, P, Z, L, *lists, n, scratch, rres, rfb, Zf, R1, R2, **extra)
        outputs.append(out)
    np.testing.assert_allclose(np.asarray(outputs[0][0]), np.asarray(outputs[1][0]), rtol=2e-5, atol=2e-8)
    if need_d21:
        a, b = np.asarray(outputs[0][1]), np.asarray(outputs[1][1])
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-8)
        np.testing.assert_array_equal(np.diag(b), np.zeros(n))
    else:
        assert outputs[0][1] is None and outputs[1][1] is None
