"""T388: the frozen T366 graded-g12 estimator (claude/t218-tie-tail-labels @5b4a1250 r1_cont/estimator_t366.est, run
unmodified; all 7 sha256 of t366_freeze_sha256.txt asserted) on T371's panel. DEDUCTIVE: no random inputs.
usage: python est_t388.py SHARD NSHARD -> results/t388_est_SHARD.json"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as MP  # noqa: E402

T0S = (1.0, 1.25, 1.5)
R1C = os.environ["T371_R1C"]; SRC = os.environ["T340_SRC"]
for line in open(os.path.join(R1C, "t366_freeze_sha256.txt")):
    h, fn = line.split()
    p = os.path.join(SRC, "m16", "approaches", fn) if "/" in fn else os.path.join(R1C, fn)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, p
sys.path.insert(0, R1C)
import estimator_t366 as E  # noqa: E402

assert E.DEDUCTIVE

if __name__ == "__main__":
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    P = MP.panel(); names = sorted(P)[shard::nsh]
    fn = os.path.join(HERE, "results", f"t388_est_{shard}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for nm in names:
        if nm in out:
            continue
        t = time.time(); r, f = E.est(P[nm], T0S)
        out[nm] = dict(est={str(k): float(v) for k, v in r.items()}, C_fpe=f / 5948.0, sec=time.time() - t)
        json.dump(out, open(fn, "w"), indent=1)
        print(nm, {k: f"{v:.3e}" for k, v in out[nm]["est"].items()}, "C %.0f" % out[nm]["C_fpe"], "%.0fs" % out[nm]["sec"], flush=True)
