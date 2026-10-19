"""T424: the frozen sources (read-only checkouts, env T424_R1C = r1_cont/ of claude/t218-tie-tail-labels @5b4a1250,
T340_SRC = a checkout of claude/t286-cheap-frontier @620eb47b). Asserts all 7 sha256 of r1_cont/t366_freeze_sha256.txt
and returns the frozen modules (t366 patches rare.walk3 = walkfix, as in every frozen run)."""
import hashlib
import os
import sys


def frozen():
    R1C = os.environ["T424_R1C"]; SRC = os.environ["T340_SRC"]; n = 0
    for line in open(os.path.join(R1C, "t366_freeze_sha256.txt")):
        h, fn = line.split()
        p = os.path.join(SRC, "m16", "approaches", fn) if "/" in fn else os.path.join(R1C, fn)
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, p; n += 1
    assert n == 7
    sys.path.insert(0, R1C)
    import estimator_t366 as E  # noqa: E402
    import t354 as S  # noqa: E402
    import t366  # noqa: E402
    return E, S, t366
