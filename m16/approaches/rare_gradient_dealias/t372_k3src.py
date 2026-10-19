"""Read-only access to the frozen K3 walk code (T364: T340c + T354 segment start + T354 walk fix, T_hi = 6) and the
models. Env T372_SRC620 = checkout of claude/t286-cheap-frontier @620eb47b, T372_R1C = checkout of
claude/t218-tie-tail-labels (r1_cont), T372_DEV3 = rare_events/ext/dev3/weights.npz. sha256s asserted below."""
import hashlib
import os
import sys

import numpy as np

SCR = "/tmp/claude-0/-home-user-clean-neurons/cd81c820-43e0-5825-aabd-78a1bd47d6c2/scratchpad"
SRC620 = os.environ.get("T372_SRC620", os.path.join(SCR, "src620"))
R1C = os.environ.get("T372_R1C", os.path.join(SCR, "r1c", "m16", "approaches", "r1_cont"))
DEV3 = os.environ.get("T372_DEV3", os.path.join(SCR, "dev3", "weights.npz"))
SHA = {os.path.join(R1C, "t340c.py"): "7e6a749d67284d4e634bfb302087b9559da4956a6d5e6a1d29dd381959ec9236",
       os.path.join(R1C, "t354.py"): "abed36c8db9dbecfe36b6e5ab41090363778a8d1485f0e7599e6f23298232dde",
       os.path.join(R1C, "walkfix.py"): "8e21b6a11c7d9151d2f02f84d1e09f5225b1e605501aa0e57aeb398a1d432d7e",
       os.path.join(SRC620, "m16/approaches/rare_events/rare.py"): "09f89a8b9eb23c589976e6f52220ca66a2cad9f24a818dbaa9353e8643cb4fa9",
       os.path.join(SRC620, "m16/approaches/closure_floor/cont_g.py"): "d34dde3699c62d5784b2051d6ce8446ca03a0180e1ef6558ef117e3525b71605"}
for p, h in SHA.items():
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, p
os.environ["T340_SRC"] = SRC620
sys.path.insert(0, R1C)
import t340c as TC  # noqa: E402
import t354 as T354  # noqa: E402
import walkfix as WF  # noqa: E402

TC.rare.walk3 = lambda model, x0, d, k, k3, tmax, max_cells=5000: WF.walk3_fixed(TC.G, model, x0, d, k, k3, tmax, max_cells)[:5]
rare, G, NT, TRI, FPE = TC.rare, TC.G, TC.NT, TC.TRI, TC.cf.FPE
CASES = (("orig", 1.25), ("dev3_s526", 3.25), ("dev3_s528", 2.75))       # pre-registered (README)


def models():
    out = {"orig": tuple(np.asarray(w, float) for w in TC.cf.lb3_models()["orig"])}
    z = np.load(DEV3); names = list(map(str, z["names"]))
    for nm in ("dev3_s526", "dev3_s528"):
        i = names.index(nm); out[nm] = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
    return out
