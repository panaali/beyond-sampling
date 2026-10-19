"""T446 (ii) locate and (iii) repair at L = 15 (pre-registered in README). usage: python lr.py MODEL
Writes data/t446/lr_<MODEL>.json."""
import json
import math
import sys
import time
import zlib

import numpy as np
import torch
from torch.func import functional_call

from est import ce_is
from s3lib import DATA, Dyck2, load, model_input, stack_info
from taus import points

L, S_SMOOTH, NW = 15, 0.25, 12


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def unflat(model, v):
    out, i = {}, 0
    for k, p in model.named_parameters():
        out[k] = v[i:i + p.numel()].reshape(p.shape); i += p.numel()
    return out


def pill_torch(model, params, y):
    """Differentiable (N, L) p_illegal for sequences y (numpy)."""
    x = torch.from_numpy(model_input(y, 4))
    lg = functional_call(model, params, (x,))
    p = torch.softmax(lg.double(), -1)
    dep, top, _ = stack_info(y)
    wrong = torch.from_numpy(np.where(top == 0, 3, 2))
    pw = p.gather(-1, wrong[..., None])[..., 0]
    return torch.where(torch.from_numpy(dep > 0), pw, p[..., 2] + p[..., 3])


def smooth_grad(model, theta, y, w, tau, chunk=4096):
    """sum_i w_i grad sigmoid((logit max_t p_ill - logit tau) / s) and the value."""
    g = torch.zeros_like(theta, dtype=torch.float64); val = 0.0
    lt = math.log(tau / (1 - tau))
    for i in range(0, len(y), chunk):
        th = theta.clone().requires_grad_(True)
        m = pill_torch(model, unflat(model, th), y[i:i + chunk]).max(1).values.clamp(1e-12, 1 - 1e-12)
        s = torch.sigmoid((torch.log(m) - torch.log1p(-m) - lt) / S_SMOOTH)
        f = (torch.from_numpy(w[i:i + chunk]) * s).sum()
        f.backward()
        g += th.grad.double(); val += float(f.detach())
    return g, val


def exact_metrics(model, params=None, taus=()):
    """Exact L = 15 closer accuracy, illegal mass and B at taus (torch.no_grad)."""
    lang = Dyck2()
    yall, wall = lang.enumerate_small(L, as_int8=True)
    an = ad = ill = 0.0; B = np.zeros(len(taus))
    with torch.no_grad():
        for i in range(0, len(yall), 100000):
            y = yall[i:i + 100000].astype(np.int64); w = wall[i:i + 100000]
            x = torch.from_numpy(model_input(y, 4))
            chunks = [functional_call(model, params, (x[j:j + 8192],)) if params else model(x[j:j + 8192]) for j in range(0, len(x), 8192)]
            p = torch.softmax(torch.cat(chunks).double(), -1).numpy()
            dep, top, _ = stack_info(y)
            wrong = np.where(top == 0, 3, 2); corr = np.where(top == 0, 2, 3)
            pw = np.take_along_axis(p, wrong[..., None], -1)[..., 0]; pc = np.take_along_axis(p, corr[..., None], -1)[..., 0]
            pi = np.where(dep > 0, pw, p[..., 2] + p[..., 3]); pos = dep > 0
            an += float((w[:, None] * ((pc > pw) & pos)).sum()); ad += float((w[:, None] * pos).sum())
            ill += float((w[:, None] * pi).sum()) / L
            mx = pi.max(1)
            for j, t in enumerate(taus):
                B[j] += float(w[mx > t].sum())
    return dict(acc=an / ad, ill=ill, B=B.tolist())


class Edited(torch.nn.Module):
    """Wraps a model with replaced flat parameters (for est.py, which calls model(x) and model.flops)."""

    def __init__(self, model, theta):
        super().__init__()
        self.m, self.p = model, unflat(model, theta.float())

    def forward(self, x):
        return functional_call(self.m, self.p, (x,))

    def flops(self, T):
        return self.m.flops(T)


def paired_acc(model, theta0, theta1, n, rng):
    """Paired closer accuracy and illegal mass differences on n fresh law samples: (d_acc, se, d_ill, se)."""
    lang = Dyck2(L=L)
    y = lang.sample(n, rng, L=L)
    dep, top, _ = stack_info(y); pos = dep > 0
    wrong = np.where(top == 0, 3, 2); corr = np.where(top == 0, 2, 3)
    res = []
    with torch.no_grad():
        for th in (theta0, theta1):
            P = []
            for j in range(0, n, 8192):
                x = torch.from_numpy(model_input(y[j:j + 8192], 4))
                P.append(torch.softmax(functional_call(model, unflat(model, th.float()), (x,)).double(), -1).numpy())
            p = np.concatenate(P)
            pw = np.take_along_axis(p, wrong[..., None], -1)[..., 0]; pc = np.take_along_axis(p, corr[..., None], -1)[..., 0]
            acc = ((pc > pw) & pos).sum(1) / np.maximum(pos.sum(1), 1)
            ill = np.where(pos, pw, p[..., 2] + p[..., 3]).mean(1)
            res.append((acc, ill))
    da = res[1][0] - res[0][0]; di = res[1][1] - res[0][1]
    # closer accuracy is a ratio of sums; use the per-sequence weighted difference
    wpos = pos.sum(1)
    dacc = float((da * wpos).sum() / wpos.sum()); se_a = float(np.std(da * wpos) * math.sqrt(n) / wpos.sum())
    return dacc, se_a, float(di.mean()), float(di.std() / math.sqrt(n))


def kl_loss(model, params, y, law):
    x = torch.from_numpy(model_input(y, 4))
    lp = torch.log_softmax(functional_call(model, params, (x,)).double(), -1)
    lw = torch.from_numpy(law)
    return (torch.xlogy(lw, lw) - lw * lp.masked_fill(lw == 0, 0)).sum(-1).mean()


def main(name):
    t0 = time.time()
    model, _ = load(name)
    for p in model.parameters():
        p.requires_grad_(False)
    lang = Dyck2(L=L)
    tau, truth, _ = points(name)[0]
    theta = flat(model).double()
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    out = dict(model=name, tau=tau, truth=truth)
    # ---- exact smoothed gradient (L = 15): sequences with logit max > logit tau - 4
    yall, wall = lang.enumerate_small(L, as_int8=True)
    mx = np.load(f"{DATA}/{name}_L15_max.npy").astype(np.float64)
    lt = math.log(tau / (1 - tau))
    keep = np.log(np.clip(mx, 1e-12, 1)) - np.log1p(-np.clip(mx, 0, 1 - 1e-12)) > lt - 4
    out["exact_grad_n_seq"] = int(keep.sum()); out["truncated_mass_bound"] = float(wall[~keep].sum() * 1 / (1 + math.exp(16)))
    g_ex, Bs_ex = smooth_grad(model, theta.float(), yall[keep].astype(np.int64), wall[keep], tau)
    out["Bs_exact"] = Bs_ex
    # ---- estimated gradient from the CE-IS final samples at 1e5 FPE
    est, cost, tw, ys, lr = ce_is(model, lang, L, tau, 1e5, rng, return_twist=True, return_samples=True)
    g_es, Bs_es = smooth_grad(model, theta.float(), ys, lr / len(ys), tau)
    out.update(B_est=est, Bs_est=Bs_es, grad_cost=cost)
    cos = float(torch.dot(g_ex, g_es) / (g_ex.norm() * g_es.norm()))
    top_ex = set(torch.topk(g_ex.abs(), NW).indices.tolist()); top_es = torch.topk(g_es.abs(), NW).indices.tolist()
    out.update(cosine=cos, top12_overlap=len(top_ex & set(top_es)))
    print(name, f"tau {tau:.4f} B {truth:.2e} Bs exact {Bs_ex:.2e} est {Bs_es:.2e} cos {cos:.3f} overlap {out['top12_overlap']}", flush=True)
    # ---- repair: Fisher diag of the excess-KL surrogate on 2e4 law samples (100 minibatches of 200)
    y = lang.sample(20000, rng, L=L); law = lang.annotate(y)["law"]
    F = torch.zeros_like(theta)
    for b in range(100):
        th = theta.float().clone().requires_grad_(True)
        kl_loss(model, unflat(model, th), y[b * 200:(b + 1) * 200], law[b * 200:(b + 1) * 200]).backward()
        F += th.grad.double() ** 2 / 100
    lam = float(F.median())
    sel = torch.topk(g_es ** 2 / (F + lam), NW).indices
    out["weights"] = sel.tolist()
    names_ = []
    for k, p in model.named_parameters():
        names_ += [k] * p.numel()
    out["weight_names"] = [names_[i] for i in sel.tolist()]

    def A_of(dS):
        th = theta.float().clone(); th = th.index_put((sel,), th[sel] + dS.float())
        return kl_loss(model, unflat(model, th), y[:4000], law[:4000])
    H = torch.autograd.functional.hessian(A_of, torch.zeros(NW, dtype=torch.float64)).double()
    lamH = float(torch.diagonal(H).abs().median()) + 1e-12
    dirn = -torch.linalg.solve(H + lamH * torch.eye(NW, dtype=torch.float64), g_es[sel])
    dirn = dirn / dirn.norm()
    nS = float(theta[sel].norm())
    chosen = None
    out["steps"] = []
    for c in (.01, .02, .05, .1, .2, .5, 1, 2):
        th1 = theta.clone(); th1[sel] += c * nS * dirn
        e1, c1 = ce_is(Edited(model, th1), lang, L, tau, 1e5, rng)
        da, sa, di, si = paired_acc(model, theta, th1, 200000, rng)
        ratio = e1 / est if est > 0 else float("nan")
        rec = dict(c=c, B_ratio_est=ratio, dacc_pp=100 * da, dacc_se_pp=100 * sa, dill=di, dill_se=si, cost=c1 + 4e5)
        out["steps"].append(rec)
        print(" step", rec, flush=True)
        if ratio <= 0.05 and 100 * da >= -0.02 and di <= 2 * si:
            chosen = c
            break
    out["chosen_c"] = chosen
    base = exact_metrics(model, None, [tau])
    out["exact_orig"] = base
    if chosen is None:
        out["result"] = "no step met the estimate-based rule"
        json.dump(out, open(f"{DATA}/lr_{name}.json", "w")); return
    th1 = theta.clone(); th1[sel] += chosen * nS * dirn
    ed = exact_metrics(model, unflat(model, th1.float()), [tau])
    out["exact_edit"] = ed

    def passes(m):
        return (m["B"][0] / base["B"][0] <= 0.1) and (100 * (m["acc"] - base["acc"]) >= -0.05) and (m["ill"] <= base["ill"])
    out["edit_pass"] = passes(ed)
    print(" edit exact", ed, "orig", base, "pass", out["edit_pass"], flush=True)
    ctr = []
    for j in range(40):
        r = np.random.default_rng(zlib.crc32(f"{name}|ctrl|{j}".encode())).standard_normal(NW)
        th2 = theta.clone(); th2[sel] += chosen * nS * torch.from_numpy(r / np.linalg.norm(r))
        m = exact_metrics(model, unflat(model, th2.float()), [tau])
        m["pass"] = passes(m); ctr.append(m)
        print(f" ctrl {j} B ratio {m['B'][0] / base['B'][0]:.3f} dacc {100 * (m['acc'] - base['acc']):.4f}pp pass {m['pass']}", flush=True)
    out["controls"] = ctr; out["controls_pass"] = int(sum(m["pass"] for m in ctr))
    out["seconds"] = time.time() - t0
    json.dump(out, open(f"{DATA}/lr_{name}.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
