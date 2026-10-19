"""Sanity tests of cert.logit_box: (1) concrete inputs reproduce the model's logits; (2) sampled sequences that
agree with a node's prefix lie inside the node's abstract box."""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mini_lm"))
from load import load, names
from langs import model_input
import cert, bab

rng = np.random.default_rng(0)
for name in [n for n in names() if n.startswith("dyck2")]:
    model, lang = load(name)
    P = cert.Params(model)
    T = 15
    y = lang.sample(64, rng, L=T)
    x = model_input(y, lang.BOS)
    al = torch.zeros(64, T, 5, dtype=torch.bool); al.scatter_(2, torch.from_numpy(x)[..., None], True)
    c, r = cert.logit_box(P, al)
    xi = torch.from_numpy(x)[:, :, None, None].expand(-1, -1, 1, 4)
    c, r = c.gather(2, xi)[:, :, 0], r.gather(2, xi)[:, :, 0]
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).double()
    e1 = (c - ref).abs().max().item(); rmax = r.max().item()
    # abstract: prefix of length k from y[0], others free; samples sharing the prefix
    worst = 0.0
    for k in (3, 8, 12):
        ys = lang.sample(20000, rng, L=T)
        ys = ys[(ys[:, :k] == y[0, :k]).all(1)]
        if len(ys) == 0:
            continue
        pre = y[:1, :k].astype(np.int8)
        st = np.zeros((1, k), np.int8); dd = 0
        for t in pre[0]:
            if t < 2: st[0, dd] = t; dd += 1
            else: dd -= 1
        top = np.array([st[0, dd - 1] if dd > 0 else 0])
        alA = bab._allowed_dyck(pre, np.array([dd]), top, T, True)
        cA, rA = cert.logit_box(P, alA)
        with torch.no_grad():
            lg = model(torch.from_numpy(model_input(ys, lang.BOS))).double()
        xs = torch.from_numpy(model_input(ys, lang.BOS))[:, :, None, None].expand(-1, -1, 1, 4)
        cS = cA.expand(len(ys), -1, -1, -1).gather(2, xs)[:, :, 0]; rS = rA.expand(len(ys), -1, -1, -1).gather(2, xs)[:, :, 0]
        viol = ((lg - (cS + rS)).clamp(min=0) + ((cS - rS) - lg).clamp(min=0)).max().item()
        rA = rS[:1]
        worst = max(worst, viol)
        wid = rA[0, k + 1:].mean().item()
    print(f"{name}: concrete |c-ref| {e1:.2e} r {rmax:.2e}; abstract max violation {worst:.2e}; mean radius at k=12 {wid:.3g}")
