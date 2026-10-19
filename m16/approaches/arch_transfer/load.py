"""Load T432 models as (kind, float64 weight dict) for netnp. Sources: weights/<tag>_s<seed>.pt (trained here),
the T421 bias-free H = 32 RNNs and ARC's M_16,10 (npz with Wi, Wh, Wo)."""
import os

import numpy as np
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
KIND = {"rnnb32": "rnn", "rnn2r32": "rnn2r", "hgru32": "hgru", "sgru32": "sgru", "ssm": "ssm", "ssmB": "ssm", "conv": "conv",
        "tfrelu": "tfrelu", "tfsoft": "tfsoft", "rnn32": "rnn", "m16": "rnn"}


def from_state(tag, sd):
    g = {k: v.detach().double().numpy() for k, v in sd.items()}
    k = KIND[tag]
    if k == "rnn":
        w = {"Wi": g["rnn.weight_ih_l0"], "Wh": g["rnn.weight_hh_l0"], "Wo": g["linear.weight"]}
        if "rnn.bias_ih_l0" in g:
            w["bh"] = g["rnn.bias_ih_l0"] + g["rnn.bias_hh_l0"]; w["bo"] = g["linear.bias"]
    elif k == "rnn2r":
        w = {"Wi1": g["r1.weight_ih_l0"], "Wh1": g["r1.weight_hh_l0"], "Wi2": g["r2.weight_ih_l0"],
             "Wh2": g["r2.weight_hh_l0"], "Wo": g["linear.weight"]}
    elif k in ("hgru", "sgru"):
        w = {"Wx": g["Wx.weight"], "bx": g["Wx.bias"], "Wh": g["Wh.weight"], "bh": g["Wh.bias"],
             "Wo": g["linear.weight"], "bo": g["linear.bias"]}
    elif k == "ssm":
        w = {"a1": np.exp(-np.exp(g["nu1"])), "a2": np.exp(-np.exp(g["nu2"])), "B1": g["B1.weight"],
             "m1": g["m1.weight"], "m2": g["m2.weight"], "B2": g["B2.weight"], "o1": g["o1.weight"], "o2": g["o2.weight"]}
    elif k == "conv":
        w = {f"c{i}": g[f"convs.{i}.weight"] for i in range(4)}
        w["h1"] = g["h1.weight"]; w["h2"] = g["h2.weight"]
    else:
        L = sum(1 for key in g if key.startswith("q."))
        w = {"e1": g["e1.weight"], "e1b": g["e1.bias"], "e2": g["e2.weight"], "e2b": g["e2.bias"],
             "out": g["out.weight"], "outb": g["out.bias"], "L": L}
        for l in range(L):
            for nm in "qkvo":
                w[f"{nm}{l}"] = g[f"{nm}.{l}.weight"]
            w[f"f1{l}"] = g[f"f1.{l}.weight"]; w[f"f1b{l}"] = g[f"f1.{l}.bias"]
            w[f"f2{l}"] = g[f"f2.{l}.weight"]; w[f"f2b{l}"] = g[f"f2.{l}.bias"]
    return k, w


def from_npz(path):
    d = np.load(path)
    return "rnn", {"Wi": d["Wi"][:, None].astype(np.float64), "Wh": d["Wh"].astype(np.float64), "Wo": d["Wo"].astype(np.float64)}


def load(tag, seed):
    if tag == "rnn32":
        return from_npz(os.path.join(HERE, "ref", f"m_32_10_cpu{seed}.npz"))
    if tag == "m16":
        return from_npz(os.path.join(HERE, "ref", "m16.npz"))
    return from_state(tag, th.load(os.path.join(HERE, "weights", f"{tag}_s{seed}.pt")))
