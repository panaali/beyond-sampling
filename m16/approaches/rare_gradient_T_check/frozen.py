"""T405: read-only access to T386's frozen family-T rule (T5/gl20) and the K3 rule, from a checkout of
claude/t386-gradient-dealias @d708a676 (env T405_T386, default <scratch>/t386). The six T386 files that define the
estimator are asserted by sha256 before import; T386's own t372_k3src.py then asserts the upstream K3 / walk code
(T372_SRC620, T372_R1C) and loads DEV3 weights (T372_DEV3). Nothing is modified; this file only sets paths.
t5(model, t0)  -> transversal.est_grad(model, t0, 5, "gl20")   (the frozen secondary config of T386, exactly)
k3(model, t0)  -> dealias.est_grad(model, t0)                  (K3 rule; reproduces T372's K3 bit for bit)"""
import hashlib
import os
import sys

import numpy as np

SCR = os.environ.get("T405_SCR", "/tmp/claude-0/-home-user-clean-neurons/54da0b56-46f9-5b60-baac-a8fae4269f96/scratchpad")
T386 = os.path.join(os.environ.get("T405_T386", os.path.join(SCR, "t386")), "m16", "approaches", "rare_gradient_dealias")
os.environ.setdefault("T372_SRC620", os.path.join(SCR, "src620"))
os.environ.setdefault("T372_R1C", os.path.join(SCR, "r1c", "m16", "approaches", "r1_cont"))
os.environ.setdefault("T372_DEV3", os.path.join(SCR, "dev3", "weights.npz"))
SHA = {"transversal.py": "623a45f15cb2c9e2694e3ab70f7a2b85238ee211fff91ce059edfe0ae986c22a",
       "dealias.py": "a50d666f5ae8f7733da65503cf9dadb7891df333a5d4bdd26ca2a84493dc38c1",
       "t372_k3src.py": "012a6568ce4a3205c737636f8556ee082431261667e3e3a4b6b815862b563d3b",
       "t372_lgrad.py": "8c726a2a94f9223f7de2510e156a273d372aa4f7dc259e1f8ef43a042eb51657",
       "t372_walk.py": "fe3d1a0364986c51fb468f34fbf705d86f902799d8cf5ff1776da9fcb1305bc3",
       "t372_rnn.py": "448027cbb56b0ef405974047b98650b5a637e63aac8e4b090e4114c8b2ade2dd"}
for f, h in SHA.items():
    assert hashlib.sha256(open(os.path.join(T386, f), "rb").read()).hexdigest() == h, f
DEV3_SHA = "62c952037e8c9ec214ee96b3dc43923a9e19cb5f137dbf16b7f6308d721591f9"   # rare_events/ext/dev3/weights.npz @t286
assert hashlib.sha256(open(os.environ["T372_DEV3"], "rb").read()).hexdigest() == DEV3_SHA
sys.path.insert(0, T386)
import dealias  # noqa: E402
import transversal  # noqa: E402

KT, V2SPEC = 5, "gl20"                     # T386 frozen secondary config T5/gl20 (results/frozen.json there)


def dev3(name):
    z = np.load(os.environ["T372_DEV3"]); i = list(map(str, z["names"])).index(name)
    return z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float)


def t5(model, t0):
    return transversal.est_grad(model, t0, KT, V2SPEC)


def k3(model, t0):
    return dealias.est_grad(model, t0)
