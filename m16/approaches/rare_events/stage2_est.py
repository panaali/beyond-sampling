"""T370 stage 2 (selection rule pre-registered 2026-10-18 23:40 + DRI amendment): truth-blind estimates on the 64 candidates
(dev3_s548-563 x {E1 @ 2.35, 2.75; E2 @ 1.25, 1.5}). Deductive (no random model inputs).
usage: python stage2_est.py k3|q16e1|q16e2 dev3_s548,dev3_s549,... -> results/t370s2/<mode>_<model>.json (one per model)
- k3:    frozen K3 (claude/t218-tie-tail-labels @e433919d, as T364 / T401), one est_seg call for E1 and E2.
- q16e1: frozen Q16 (@6441c0c5 t370.est_e1(model, (2.35, 2.75), 16)), one call for both t0.
- q16e2: frozen Q16 (@6441c0c5 t370.est_e2(model, t, 16)) at t = 1.25 and 1.5."""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S0 = "/tmp/claude-0/-home-user-clean-neurons/18e70fa5-7ff5-5fac-af53-4fea2351128f/scratchpad"
OUT = os.path.join(HERE, "results", "t370s2"); os.makedirs(OUT, exist_ok=True)
T0S = (2.35, 2.75); T2S = (1.25, 1.5); FPE = 5948.0
mode, names = sys.argv[1], sys.argv[2].split(",")
z = np.load(os.path.join(HERE, "ext", "dev3", "weights.npz")); allnames = list(map(str, z["names"]))


def sha(d, files):
    for f, h in files.items():
        assert hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() == h, f
    return files


if mode == "k3":
    R1C = os.path.join(S0, "r1c", "m16", "approaches", "r1_cont")
    SHA = sha(R1C, {"t340c.py": "7e6a749d67284d4e634bfb302087b9559da4956a6d5e6a1d29dd381959ec9236",
                    "t354.py": "abed36c8db9dbecfe36b6e5ab41090363778a8d1485f0e7599e6f23298232dde",
                    "walkfix.py": "8e21b6a11c7d9151d2f02f84d1e09f5225b1e605501aa0e57aeb398a1d432d7e"})
    commit = "e433919d"
    sys.path.insert(0, HERE)
    from deep import S  # noqa: E402  (deep patches walk3 with walkfix)
else:
    R1C = os.path.join(S0, "r1q16", "m16", "approaches", "r1_cont")
    SHA = sha(R1C, {"t370.py": "155ce0f643c69e8a7f72b17f5ad92bfd44372e252aee8b60a91864c6d1531e2d",
                    "t366.py": "997a1b60052757b3048c9baf846245654e8bdc6d2c75f6e5224a746b18064774",
                    "t340c.py": "7e6a749d67284d4e634bfb302087b9559da4956a6d5e6a1d29dd381959ec9236",
                    "t354.py": "abed36c8db9dbecfe36b6e5ab41090363778a8d1485f0e7599e6f23298232dde",
                    "walkfix.py": "8e21b6a11c7d9151d2f02f84d1e09f5225b1e605501aa0e57aeb398a1d432d7e",
                    "cont_e.py": "50dd1c9b85512e23f037171d19de03527f8f5989758cfb74d93b441de48fda2b"})
    SRC = os.path.join(S0, "src620", "m16", "approaches")
    SHA.update(sha(SRC, {"rare_events/rare.py": "09f89a8b9eb23c589976e6f52220ca66a2cad9f24a818dbaa9353e8643cb4fa9",
                         "closure_floor/cont_g.py": "d34dde3699c62d5784b2051d6ce8446ca03a0180e1ef6558ef117e3525b71605"}))
    commit = "6441c0c5"
    os.environ["T340_SRC"] = os.path.join(S0, "src620")
    sys.path.insert(0, R1C)
    import t370  # noqa: E402

for nm in names:
    assert nm in ["dev3_s%d" % s for s in range(548, 564)], nm
    i = allnames.index(nm); m = tuple(z[k][i].astype(np.float64) for k in ("Wi", "Wh", "Wo"))
    t = time.time()
    if mode == "k3":
        r1, r2, f = S.est_seg(m, 3, t0s=T0S, t2s=T2S, tmax=6.0)
        out = dict(E1={str(k): float(v) for k, v in r1.items()}, E2={str(k): float(v) for k, v in r2.items()}, C=f / FPE)
    elif mode == "q16e1":
        r1, f = t370.est_e1(m, T0S, 16)
        out = dict(E1={str(k): float(v) for k, v in r1.items()}, C=f / FPE)
    else:
        out = dict(E2={}, C_t={})
        for t2 in T2S:
            v, f = t370.est_e2(m, t2, 16); out["E2"][str(t2)] = float(v); out["C_t"][str(t2)] = f / FPE
        out["C"] = sum(out["C_t"].values())
    out.update(model=nm, mode=mode, commit=commit, sha256=SHA, sec=time.time() - t)
    print(json.dumps({k: out[k] for k in out if k != "sha256"}), flush=True)
    json.dump(out, open(os.path.join(OUT, f"{mode}_{nm}.json"), "w"), indent=1)
