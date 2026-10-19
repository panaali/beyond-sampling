"""T366 frozen estimator: T364's K3 estimator (T340c + walk fix + T354 segment start, T_hi = 6) with the graded g12 rule
clustered at the triple tie (t366.py). E1(t0) = {pick = p3 AND g23 > t0}.
Requires env T340_SRC = a checkout of claude/t286-cheap-frontier @620eb47b (read-only: rare_events/rare.py and
closure_floor/cont_g.py, pinned; sha256s in the README T366 freeze note).
est(model, t0s) -> ({t0: estimate}, FLOPs as executed). Deductive: no random draws."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t366  # noqa: E402

DEDUCTIVE = True


def est(model, t0s):
    return t366.est_e1(model, tuple(t0s))
