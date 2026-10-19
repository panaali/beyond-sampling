"""T423 shared helpers: frozen estimator import (sha256-asserted), T421 models, result files."""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SS = os.path.join(HERE, "../scale_study")          # T421 folder, checked out untracked by refresh.sh
SHA = {"frozen/estimator.py": "fb7902e12240acd86dc6cd18786018abe73201a1996357efead379bd487fa867",
       "frozen/core.py": "48dbba49ab13367615e63a396d56f506bba9f8adda76ced4886a6023ecef8283",
       "frozen/walk.py": "8cdf611a37348f87b4c6947710ffd52f3f89286d52262884870d5f763066344b",
       "frozen/hom.py": "a91f019fca215b5b95996f77ff514811679ffda645112f010a0ec780a78b446d",
       "walk_nb.py": "bbc6cc992c7c6263227d4420757453a9999922c12d226ccdf1b1c2c16b99279c",
       "sampler_t393.py": "c4d5d75ffd6d6607f4b580f99bbea763967e8551eeb1901d83534f7f69d300a8"}
STATS = {"lines": 0, "cells": 0, "cap": 0, "maxcells": 0}


def _check():
    for f, h in SHA.items():
        assert hashlib.sha256(open(f"{HERE}/{f}", "rb").read()).hexdigest() == h, f


def frozen(compiled=True):
    """Frozen T10 estimator module; with compiled=True its walk is T393's walk_nb in fast mode (counting wrapper
    records lines, cells and cap hits; it does not change any value)."""
    _check()
    sys.path.insert(0, f"{HERE}/frozen"); sys.path.insert(0, HERE)
    import estimator as E  # noqa: E402
    import walk_nb as WN  # noqa: E402
    if not hasattr(E.W, "_numpy_walk"):
        E.W._numpy_walk = E.W.walk
    WN.MODE["fast"] = True; WN.MODE["fm"] = False
    base = WN.walk if compiled else E.W._numpy_walk

    def counted(*a, **k):
        mass, cost, cells = base(*a, **k)
        mc = k.get("max_cells", 4000)
        STATS["lines"] += len(cells); STATS["cells"] += int(cells.sum())
        STATS["cap"] += int((cells >= mc).sum()); STATS["maxcells"] = max(STATS["maxcells"], int(cells.max(initial=0)))
        return mass, cost, cells
    E.W.walk = counted
    return E


def registry():
    return json.load(open(os.path.join(SS, "registry.json")))["models"]


def load(tag):
    d = np.load(os.path.join(SS, "weights", f"{tag}.npz"))
    return tuple(np.ascontiguousarray(d[k], dtype=np.float64) for k in ("Wi", "Wh", "Wo"))


def sha_npz(tag):
    return hashlib.sha256(open(os.path.join(SS, "weights", f"{tag}.npz"), "rb").read()).hexdigest()


def jl_read(name):
    p = f"{HERE}/{name}"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def jl_add(name, row):
    with open(f"{HERE}/{name}", "a") as f:
        f.write(json.dumps(row) + "\n")
