"""Zonotope-style (affine-form) bounds over token sets, with the input domain a PRODUCT OF SIMPLICES (T447).

Each abstract input position a holds a token from a set S_a; it is represented by a one-hot indicator
lambda_a in the simplex over S_a. Every quantity is an affine form  C + sum_{a, s} G[a, s] lambda_{a, s}  plus a box
of radius R. Linear layers are exact on affine forms; the bound of an affine form over the product of simplices is
exact (sum over positions of the min / max coefficient over allowed tokens). ReLU uses the DeepZ relaxation (Singh
et al. 2018), with the new error term kept in the box; products (attention scores and weights) use the standard
affine-form product (first-order terms exact, the quadratic remainder bounded by intervals; as in DeepT, Bonaert
et al. 2021, which is McCormick-equivalent for a single product). As in cert.py, the state of position i is kept per
value of its own token (exact branching); other positions read its hull over its own token, which turns the own
token into the generator lambda_i (first-order exact) plus a box for the second-order remainder.

This file holds the RNN (relu_rnn_closure) and the transformer (tf_closure) closure tests used by bab_zono.py.
"""
import math
import torch

from cert import _infl, BIG, HARD_TIE, HARD_ETA

torch.set_default_dtype(torch.float64)
FLOPS = [0]


def bounds(C, G, R, tokmask):
    """C, R [..., D]; G [..., D, A, 4]; tokmask [N, A, 4] broadcast on the leading axis -> lo, hi [..., D]."""
    tm = tokmask.reshape(tokmask.shape[0], *([1] * (G.dim() - 3)), *tokmask.shape[1:])
    gmin = torch.where(tm, G, BIG).amin(-1)
    gmax = torch.where(tm, G, -BIG).amax(-1)
    gmin = torch.where(gmin >= BIG, 0.0, gmin).sum(-1)
    gmax = torch.where(gmax <= -BIG, 0.0, gmax).sum(-1)
    return C - R + gmin, C + R + gmax


def lin(C, G, R, W, b=None):
    """x @ W.T + b on an affine form; G [..., D, A, 4]."""
    FLOPS[0] += 2 * C[..., 0].numel() * W.shape[0] * W.shape[1] * (1 + G.shape[-1] * G.shape[-2]) + 2 * R.numel() * W.shape[0]
    C2 = C @ W.T
    if b is not None:
        C2 = C2 + b
    G2 = torch.einsum("...dag,od->...oag", G, W)
    return C2, G2, _infl(C2, R @ W.abs().T)


def relu(C, G, R, tokmask):
    lo, hi = bounds(C, G, R, tokmask)
    alpha = torch.where(hi <= 0, 0.0, torch.where(lo >= 0, 1.0, hi / (hi - lo).clamp(min=1e-300)))
    beta = torch.where((lo < 0) & (hi > 0), -alpha * lo / 2, 0.0)
    C2 = alpha * C + beta
    return C2, alpha[..., None, None] * G, _infl(C2, alpha * R + beta)


def hull_own(C, G, R, own_mask, a, tokmask):
    """Union over the own token s (axis -2 of C, R; -4 of G) -> one affine form with lambda_a for that choice.
    C, R [N, P, S, D]; G [N, P, S, D, A, 4]; own_mask [N, P, S] (the allowed own tokens); a [P] generator index of
    each position."""
    N, Pn, S, D = C.shape
    om = own_mask[..., None]
    w = om.double() / om.double().sum(2, keepdim=True)
    Gbar = (G * w[..., None, None]).sum(2)                       # [N, P, D, A, 4]
    dev = G - Gbar[:, :, None]
    dlo, dhi = bounds(torch.zeros_like(C), dev, R, tokmask)        # [N, P, S, D]
    lo = torch.where(om, dlo, BIG).amin(2)
    hi = torch.where(om, dhi, -BIG).amax(2)
    C2, R2 = (lo + hi) / 2, (hi - lo) / 2
    G2 = Gbar.clone()
    for p in range(Pn):
        if True:
            # lambda_{a, s} carries the branch centre C_s (sum_s lambda_s = 1 replaces the constant)
            G2[:, p, :, a[p], :S] = torch.where(own_mask[:, p, None, :], C[:, p].transpose(1, 2), 0.0)[..., :4]
    return C2, G2, _infl(C2, R2)


def mul(XC, XG, XR, YC, YG, YR, tokmask):
    """Elementwise product of two affine forms (same shape): first-order exact, remainder in the box."""
    xlo, xhi = bounds(torch.zeros_like(XC), XG, torch.zeros_like(XR), tokmask)
    ylo, yhi = bounds(torch.zeros_like(YC), YG, torch.zeros_like(YR), tokmask)
    # X = XC + gx + ex, Y = YC + gy + ey with gx in [xlo, xhi], |ex| <= XR
    C = XC * YC
    G = XC[..., None, None] * YG + YC[..., None, None] * XG
    xa = torch.maximum(xlo.abs(), xhi.abs()) + XR
    ya = torch.maximum(ylo.abs(), yhi.abs()) + YR
    # (gx + ex)(gy + ey) + XC ey + YC ex
    p = torch.stack([xlo * ylo, xlo * yhi, xhi * ylo, xhi * yhi])
    plo, phi = p.amin(0), p.amax(0)
    rem_c = (plo + phi) / 2
    rem_r = (phi - plo) / 2 + XR * ya + YR * (torch.maximum(xlo.abs(), xhi.abs())) + XC.abs() * YR + YC.abs() * XR
    return C + rem_c, G, _infl(C + rem_c, rem_r)


# ----------------------------------------------------------------------------------------------- RNN
def rnn_closure(P, h0, allowed_ab, thr_fn):
    """h0 [N, H]: exact hidden state after the concrete prefix (input position k). allowed_ab [N, A, 4]: allowed input
    tokens at abstract positions k+1..k+A. thr_fn(a, lo, hi) -> ok [N] for logit bounds lo, hi [N, 4(own), 4] at abstract
    step a. Returns ok [N, A]."""
    sd = P.sd
    W, b, U = sd["W.weight"], sd["W.bias"], sd["U.weight"][:4]
    Wo, bo = sd["out.weight"], sd["out.bias"]
    N, A, _ = allowed_ab.shape
    H = W.shape[0]
    C = h0; G = torch.zeros(N, H, A, 4); R = torch.zeros(N, H)
    oks = []
    for a in range(A):
        pc, pg, pr = lin(C, G, R, W, b)
        Cs = pc[:, None] + U[None]                                   # [N, 4, H]
        Gs = pg[:, None].expand(N, 4, H, A, 4)
        Rs = pr[:, None].expand(N, 4, H)
        Cs, Gs, Rs = relu(Cs, Gs, Rs, allowed_ab)
        oc, og, orr = lin(Cs, Gs, Rs, Wo, bo)
        lo, hi = bounds(oc, og, orr, allowed_ab)
        oks.append(thr_fn(a, lo, hi))
        C, G, R = hull_own(Cs[:, None], Gs[:, None], Rs[:, None], allowed_ab[:, a][:, None], torch.tensor([a]),
                           allowed_ab)
        C, G, R = C[:, 0], G[:, 0], R[:, 0]
    return torch.stack(oks, 1)


# ------------------------------------------------- RNN, DeepZ with ReLU errors kept as generators (M2b)
def rnn_closure_eg(P, h0, allowed_ab, thr_fn, max_err=4096):
    """As rnn_closure, but each crossing ReLU adds an error generator eps in [-1, 1] (Singh et al. 2018 DeepZ), so the
    recurrent matrix acts on errors relationally. Own-token branches get separate error variables; the hull keeps
    their mean and boxes the deviation. Same interface."""
    sd = P.sd
    W, b, U = sd["W.weight"], sd["W.bias"], sd["U.weight"][:4]
    Wo, bo = sd["out.weight"], sd["out.bias"]
    N, A, _ = allowed_ab.shape
    H = W.shape[0]
    C = h0; G = torch.zeros(N, H, A, 4); E = torch.zeros(N, H, 0); R = torch.zeros(N, H)
    oks = []
    for a in range(A):
        pc = C @ W.T + b
        pg = torch.einsum("nhag,oh->noag", G, W)
        pe = torch.einsum("nhe,oh->noe", E, W)
        pr = _infl(pc, R @ W.abs().T)
        FLOPS[0] += 2 * N * H * H * (1 + A * 4 + E.shape[-1]) + 2 * N * H * H
        Cs = pc[:, None] + U[None]                                     # [N, 4, H]
        lo, hi = bounds(Cs, pg[:, None].expand(N, 4, H, A, 4), pr[:, None].expand(N, 4, H), allowed_ab)
        er = pe.abs().sum(-1)[:, None]
        lo, hi = lo - er, hi + er
        alpha = torch.where(hi <= 0, 0.0, torch.where(lo >= 0, 1.0, hi / (hi - lo).clamp(min=1e-300)))
        beta = torch.where((lo < 0) & (hi > 0), -alpha * lo / 2, 0.0)
        Cs = alpha * Cs + beta
        Gs = alpha[..., None, None] * pg[:, None]
        Es_old = alpha[..., None] * pe[:, None]                        # [N, 4, H, E]
        Es_new = torch.diag_embed(beta)                                # [N, 4, H, H]  new eps per (branch, neuron)
        Rs = _infl(Cs, alpha * pr[:, None])
        # logits per own token
        oc = Cs @ Wo.T + bo
        og = torch.einsum("nshag,oh->nsoag", Gs, Wo)
        oe = torch.cat([torch.einsum("nshe,oh->nsoe", Es_old, Wo), torch.einsum("nshe,oh->nsoe", Es_new, Wo)], -1)
        orr = _infl(oc, Rs @ Wo.abs().T)
        llo, lhi = bounds(oc, og, orr, allowed_ab)
        oer = oe.abs().sum(-1)
        oks.append(thr_fn(a, llo - oer, lhi + oer))
        # hull over own token a: shared old errors -> mean + deviation box; new branch errors -> mean (the 4 branches'
        # new eps are kept as 4 x H separate variables: concat over branches, scaled by the branch weight bound in box)
        om = allowed_ab[:, a]                                           # [N, 4]
        w = om.double() / om.double().sum(1, keepdim=True)
        Gbar = (Gs * w[:, :, None, None, None]).sum(1)
        Ebar = (Es_old * w[:, :, None, None]).sum(1)
        # new errors: variable eps_{s, h}; lambda_s * beta_{s,h} eps_{s,h} in [-beta, beta]: keep as generator with
        # coefficient beta (|lambda_s| <= 1 keeps the bound) -> sound since eps is free in [-1, 1]
        Enew = Es_new.permute(0, 2, 1, 3).reshape(N, H, 4 * H) * om.repeat_interleave(H, 1)[:, None].double()
        dev_lo, dev_hi = bounds(torch.zeros_like(Cs), Gs - Gbar[:, None], Rs, allowed_ab)
        de = (Es_old - Ebar[:, None]).abs().sum(-1)
        dev_lo, dev_hi = dev_lo - de, dev_hi + de
        lo2 = torch.where(om[..., None], dev_lo, BIG).amin(1)
        hi2 = torch.where(om[..., None], dev_hi, -BIG).amax(1)
        C = (lo2 + hi2) / 2
        R = _infl(C, (hi2 - lo2) / 2)
        G = Gbar.clone()
        G[:, :, a, :] = torch.where(om[:, None, :], Cs.transpose(1, 2), 0.0)
        E = torch.cat([Ebar, Enew], -1)
        if E.shape[-1] > max_err:      # fold the smallest error columns into the box
            mag = E.abs().sum((0, 1))
            keep = mag.argsort(descending=True)[:max_err]
            drop = torch.ones(E.shape[-1], dtype=torch.bool); drop[keep] = False
            R = R + E[..., drop].abs().sum(-1)
            E = E[..., keep]
    return torch.stack(oks, 1)
