"""T421 registry access without touching T421's folder: registry.json and weights npz are read from the remote branch
claude/t421-scale-study by `git show` (weights cached in data/t421/, git-ignored) and the npz sha256 is checked
against the registry. load(tag) -> (Wi (H,), Wh (H, H), Wo (T, H)) float64."""
import hashlib
import io
import json
import os
import subprocess

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
BR = "origin/claude/t421-scale-study"; DIR = "m16/approaches/scale_study"
CACHE = os.path.join(ROOT, "data", "t421")


def _show(path, rev=BR):
    return subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{DIR}/{path}"], capture_output=True, check=True).stdout


def fetch():
    subprocess.run(["git", "-C", ROOT, "fetch", "-q", "origin", "claude/t421-scale-study"], check=True)
    return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", BR], capture_output=True, text=True).stdout.strip()


def registry():
    return json.loads(_show("registry.json"))


def load(tag):
    rec = {m["tag"]: m for m in registry()["models"]}[tag]; os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{tag}.npz")
    if not os.path.exists(p):
        open(p, "wb").write(_show(rec["npz"]))
    d = np.load(p)
    Wi, Wh, Wo = (d[k].astype(np.float64) for k in ("Wi", "Wh", "Wo"))
    h = hashlib.sha256(b"".join(np.ascontiguousarray(d[k]).tobytes() for k in ("Wi", "Wh", "Wo"))).hexdigest()
    assert h == rec["sha256_npz_weights"], tag
    return (Wi, Wh, Wo), rec, h
