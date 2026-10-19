"""T403 model panel: DEV3 retrains dev3_s564..dev3_s599, the only DEV3 seeds that appear in no rare-event INDEX row,
README or result file (excluded: s500-s547 used by T340-T388; s548-s563 reserved for T370 stage 2). Weights:
data/dev3/weights.npz = claude/t286-cheap-frontier m16/approaches/rare_events/ext/dev3/weights.npz (sha256 asserted)."""
import hashlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")
SHA = "62c952037e8c9ec214ee96b3dc43923a9e19cb5f137dbf16b7f6308d721591f9"
NAMES = [f"dev3_s{s}" for s in range(564, 600)]
_Z = None


def load(name, check_only=False):
    """check_only=True allows an excluded (already used) DEV3 model, for code checks only."""
    global _Z
    assert check_only or name in NAMES, name
    if _Z is None:
        assert hashlib.sha256(open(WEIGHTS, "rb").read()).hexdigest() == SHA
        _Z = np.load(WEIGHTS)
    names = [str(s) for s in _Z["names"]]; i = names.index(name)
    return tuple(_Z[k][i].astype(np.float64) for k in ("Wi", "Wh", "Wo"))
