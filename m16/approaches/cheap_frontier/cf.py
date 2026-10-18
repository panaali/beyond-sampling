"""Shared helpers for T286: LB3 panel, truths, FPE convention (T196: 5948 FLOPs, multiply-add = 2) and the
own-cost figure of merit F = RMS^2 * max(C, 1000) / mean p(1-p) (Directive 8 amendment 2)."""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "m16"))
FPE = 5948.0
FLOOR = 1000.0
T, H = 10, 16
TRIPLES = np.array(list(itertools.permutations(range(T), 3)))          # lexicographic, as in rank_exact/observer
TRIPLE_INDEX = {tuple(t): j for j, t in enumerate(TRIPLES)}
GROUPS = ("orig", "seed", "s0.0025", "s0.005", "s0.01", "s0.02")


def cyclic_subset(stride):
    """Position-balanced subset of the 720 (rank1, rank2, rank3) triples: (p, p+a, p+b) mod 10 for every p and
    every stride-th offset pair (a, b). Each position is rank 1, 2 and 3 equally often. Returns indices into TRIPLES."""
    pairs = [(a, b) for a in range(1, T) for b in range(1, T) if a != b]     # 72
    idx = [TRIPLE_INDEX[(p, (p + a) % T, (p + b) % T)] for (a, b) in pairs[::stride] for p in range(T)]
    return np.array(sorted(idx))


def lb3_models():
    import leaderboard as L
    return L.models(16, 10)


def lb3_truths():
    return json.load(open(os.path.join(ROOT, "results", "leaderboard_truths.json")))


def group_of(name):
    if name.startswith(("fused_", "torch_")):
        return name.split("_")[0]
    return "orig" if name == "orig" else "seed" if name.startswith("seed") else name.rsplit("_", 1)[0]


def score(rows, truth, groups=GROUPS, nboot=2000):
    """rows: list of dict(model, est, cost_fpe). -> {group: dict(C, bias, rms, F, F_ci)}"""
    out = {}
    for g in groups:
        sel = [r for r in rows if group_of(r["model"]) == g]
        if not sel:
            continue
        e = np.array([r["est"] - truth[r["model"]]["acc"] for r in sel])
        se2 = np.array([truth[r["model"]]["se"] ** 2 for r in sel])
        pq = np.array([truth[r["model"]]["acc"] * (1 - truth[r["model"]]["acc"]) for r in sel])
        C = float(np.mean([r["cost_fpe"] for r in sel])); Ce = max(C, FLOOR)
        mse = max(float(np.mean(e ** 2 - se2)), 0.0)
        F = mse * Ce / pq.mean()
        rng = np.random.default_rng(0); bs = []
        for _ in range(nboot):
            i = rng.integers(0, len(sel), len(sel))
            bs.append(max(np.mean(e[i] ** 2 - se2[i]), 0) * Ce / pq[i].mean())
        out[g] = dict(n=len(sel), C=C, bias_pp=100 * e.mean(), rms_pp=100 * np.sqrt(mse), F=F,
                      F_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return out


DEV2_GROUPS = ("fused", "torch")


def dev2_models():
    z = np.load(os.path.join(HERE, "ext", "dev2", "weights.npz"))
    return {str(n): (z["Wi"][i], z["Wh"][i], z["Wo"][i]) for i, n in enumerate(z["names"])}


def dev2_truths():
    """Official T284 truths if present (ext/dev2/truths.json), else this folder's provisional 4M-sample truths."""
    for p in ("truths.json", "truths_provisional.json"):
        f = os.path.join(HERE, "ext", "dev2", p)
        if os.path.exists(f):
            return json.load(open(f))
    raise FileNotFoundError("no DEV2 truths")
