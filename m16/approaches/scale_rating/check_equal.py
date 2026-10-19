"""T427 gate: walkgen.Walker(16, 10) must reproduce the frozen T407 wrapper (estimator_t366 functions, sha256 asserted
there) on M_16,10: estimates, flag shares and C, stored in claude/t286-cheap-frontier rare_events/results/t407_est.json.
Also asserts the sha256s of the frozen sources this file generalises. -> results/check_equal.json"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "m16"))
import walkgen  # noqa: E402

FROZEN = {("5b4a1250", "m16/approaches/r1_cont/estimator_t366.py"): "6bd6822e51bd06e807dc7c51baefee04931ad49a86f168c7ed6136473350dffe",
          ("5b4a1250", "m16/approaches/r1_cont/t366.py"): "997a1b60052757b3048c9baf846245654e8bdc6d2c75f6e5224a746b18064774",
          ("5b4a1250", "m16/approaches/r1_cont/t340c.py"): "7e6a749d67284d4e634bfb302087b9559da4956a6d5e6a1d29dd381959ec9236",
          ("5b4a1250", "m16/approaches/r1_cont/t354.py"): "abed36c8db9dbecfe36b6e5ab41090363778a8d1485f0e7599e6f23298232dde",
          ("5b4a1250", "m16/approaches/r1_cont/walkfix.py"): "8e21b6a11c7d9151d2f02f84d1e09f5225b1e605501aa0e57aeb398a1d432d7e"}


def git_show(rev, path):
    return subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{path}"], capture_output=True, check=True).stdout


if __name__ == "__main__":
    for (rev, p), h in FROZEN.items():
        assert hashlib.sha256(git_show(rev, p)).hexdigest() == h, p
    ref = json.loads(git_show("687a539d", "m16/approaches/rare_events/results/t407_est.json"))
    from common import Wi, Wh, Wo
    W = walkgen.Walker(16, 10); t0s = tuple(float(k) for k in ref["est"]); t = time.time()
    r, share, f, itmax = W.est_with_flag((Wi, Wh, Wo), t0s)
    out = dict(sec=time.time() - t, C=f / 5948, C_ref=ref["C"], itmax=itmax,
               rel_est={str(k): r[k] / ref["est"][str(k)] - 1 for k in t0s},
               diff_share={str(k): share[k] - ref["flag_share_g12_lt_0p1"][str(k)] for k in t0s})
    out["pass"] = bool(max(abs(v) for v in out["rel_est"].values()) < 1e-12 and max(abs(v) for v in out["diff_share"].values()) < 1e-12
                       and abs(out["C"] / out["C_ref"] - 1) < 1e-12)
    print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "results", "check_equal.json"), "w"), indent=1)
