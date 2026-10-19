"""Interval (IBP) bounds for the T433 mini-LMs over sets of token sequences (T447, stream S4).

A batch of N prefix nodes is given by a mask allowed [N, T, V_in] of the input tokens each position may hold
(one-hot at known positions, a set at abstract positions). `logit_box` returns, for every position i and every
allowed own token t_i, a centre / radius box of the logits at i that encloses the logits of EVERY sequence the sets
allow with x_i = t_i.

Representation ("own-token union"): the state of position i is kept separately for each value of its own input
token (a union of up to V_in boxes, one per token), and is boxed over its own token (the hull of the union) only
where OTHER positions read it (keys, values, the RNN's recurrent state). Self-attention to position i itself uses the
exact own-token branch. Everything else is plain interval bound propagation (IBP; Gowal et al. 2018, interval
arithmetic Moore 1966) in centre-radius form; average-hard attention is bounded by the hull of the candidate values
(every position whose score can reach the row maximum); softmax by the weight box intersected with the value hull.
"""
import math
import torch

torch.set_default_dtype(torch.float64)
REL = 1e-12          # radius inflation per operation (>> float64 rounding of <= 64-term dot products)
HARD_TIE = 1e-6      # the harness's tie tolerance for hard attention
HARD_ETA = 1e-4      # extra slack covering float32 score error in the harness's evaluation
FLOPS = [0]
BIG = 1e300


def _infl(c, r):
    return r + REL * (c.abs() + r)


def lin(c, r, W, b=None):
    """Box through x @ W.T + b. W [out, in]."""
    FLOPS[0] += 4 * c[..., 0].numel() * W.shape[0] * W.shape[1]
    cc = c @ W.T
    if b is not None:
        cc = cc + b
    return cc, _infl(cc, r @ W.abs().T)


def bil(ac, ar, bc, br):
    """Box through the matrix product A @ B of two boxes (centre-radius product)."""
    FLOPS[0] += 8 * ac[..., 0].numel() * ac.shape[-1] * bc.shape[-1]
    c = ac @ bc
    r = ac.abs() @ br + ar @ (bc.abs() + br)
    return c, _infl(c, r)


def mul(ac, ar, bc, br):
    c = ac * bc
    return c, _infl(c, ac.abs() * br + ar * (bc.abs() + br))


def relu(c, r):
    lo, hi = (c - r).clamp(min=0), (c + r).clamp(min=0)
    return (lo + hi) / 2, (hi - lo) / 2


def hull(c, r, allowed):
    """Hull over the own-token axis: c, r [N, T, V, d], allowed [N, T, V] -> [N, T, d]."""
    a = allowed[..., None]
    lo = torch.where(a, c - r, BIG).amin(2)
    hi = torch.where(a, c + r, -BIG).amax(2)
    return (lo + hi) / 2, (hi - lo) / 2


class Params:
    """float64 copy of a loaded model's weights."""

    def __init__(self, model):
        self.cfg = dict(model.cfg)
        self.sd = {k: v.detach().double() for k, v in model.state_dict().items()}
        self.arch = self.cfg["arch"]
        self.TE = None   # optional per-(node, position, slot) token embeddings [N, T, S, d] (concrete fast path)


def _res_lin(P, rest, W, b, T):
    """W applied to the residual h = tok[t] + pos[i] + rest[i, t], per (position, own token): [N, T, V, out]."""
    tw = P.TE[0] @ W.T if P.TE is not None else P.sd["tok.weight"] @ W.T   # [V, out] (or [N, T, S, out]) exact per token
    pw = P.sd["pos.weight"][:T] @ W.T          # [T, out]
    c = (tw if P.TE is not None else tw[None, None]) + pw[None, :, None]
    r = torch.zeros_like(c)
    if rest is not None:
        rc, rr = lin(rest[0], rest[1], W)
        c, r = c + rc, rr
    else:
        c = c.expand(1, *c.shape[1:])
    if b is not None:
        c = c + b
    return c, _infl(c, r)


def _attn(P, l, rest, allowed, T):
    sd, pre, attn = P.sd, f"blocks.{l}.", P.arch
    d = P.cfg["d"]
    N, _, V = allowed.shape
    vc, vr = _res_lin(P, rest, sd[pre + "v.weight"], None, T)       # [N|1, T, V, d]
    vc, vr = vc.expand(N, -1, -1, -1), vr.expand(N, -1, -1, -1)
    vhc, vhr = hull(vc, vr, allowed)                                 # [N, T, d]
    strict = torch.ones(T, T, dtype=torch.bool).tril(-1)            # j < i
    eye = torch.eye(T, dtype=torch.bool)
    if attn == "pos":
        A = sd[pre + "A"][:T, :T].masked_fill(~(strict | eye), float("-inf"))
        w = torch.softmax(A, -1)                                     # [T, T]
        FLOPS[0] += 4 * N * T * T * d
        wo = w.masked_fill(eye, 0)
        c = (wo @ vhc)[:, :, None] + torch.diagonal(w)[None, :, None, None] * vc
        r = (wo @ vhr)[:, :, None] + torch.diagonal(w)[None, :, None, None] * vr
        return c, _infl(c, r)
    qc, qr = _res_lin(P, rest, sd[pre + "q.weight"], None, T)
    kc, kr = _res_lin(P, rest, sd[pre + "k.weight"], None, T)
    qc, qr = qc.expand(N, -1, -1, -1), qr.expand(N, -1, -1, -1)
    kc, kr = kc.expand(N, -1, -1, -1), kr.expand(N, -1, -1, -1)
    khc, khr = hull(kc, kr, allowed)
    sc_, sr_ = bil(qc.reshape(N, T * V, d), qr.reshape(N, T * V, d), khc.transpose(1, 2), khr.transpose(1, 2))
    sc_, sr_ = sc_.reshape(N, T, V, T), sr_.reshape(N, T, V, T)
    dc, dr = mul(qc, qr, kc, kr)                                     # diagonal (same own token)
    dc, dr = dc.sum(-1), dr.sum(-1)                                  # [N, T, V]
    FLOPS[0] += 8 * N * T * V * d
    E = eye[None, :, None, :]
    sc_ = torch.where(E, dc[..., None], sc_) / math.sqrt(d)
    sr_ = torch.where(E, dr[..., None], sr_) / math.sqrt(d)
    M = (strict | eye)[None, :, None, :]                             # [1, T(i), 1, T(j)]
    slo, shi = sc_ - sr_, sc_ + sr_
    if attn in ("relu", "linear"):
        if attn == "relu":
            slo, shi = slo.clamp(min=0), shi.clamp(min=0)
        cnt = torch.arange(1, T + 1, dtype=torch.float64)[None, :, None, None]
        slo, shi = slo.masked_fill(~M, 0) / cnt, shi.masked_fill(~M, 0) / cnt
        wc, wr = (slo + shi) / 2, (shi - slo) / 2
        oc, orr = bil(wc.masked_fill(E, 0).reshape(N, T * V, T), wr.masked_fill(E, 0).reshape(N, T * V, T), vhc, vhr)
        oc, orr = oc.reshape(N, T, V, d), orr.reshape(N, T, V, d)
        wdc = torch.diagonal(wc, dim1=1, dim2=3).permute(0, 2, 1)[..., None]  # [N, T, V, 1]
        wdr = torch.diagonal(wr, dim1=1, dim2=3).permute(0, 2, 1)[..., None]
        ec, er = mul(wdc, wdr, vc, vr)
        return oc + ec, _infl(oc + ec, orr + er)
    # values seen by row (i, t_i) at column j: hull for j < i, own branch for j = i
    vlo_j = (vhc - vhr)[:, None, None].expand(N, T, V, T, d)
    vhi_j = (vhc + vhr)[:, None, None].expand(N, T, V, T, d)
    Ed = eye[None, :, None, :, None]
    vlo_j = torch.where(Ed, (vc - vr)[:, :, :, None], vlo_j)
    vhi_j = torch.where(Ed, (vc + vr)[:, :, :, None], vhi_j)
    FLOPS[0] += 2 * N * T * V * T * d
    if attn == "hard":
        best_lo = slo.masked_fill(~M, -BIG).amax(-1, keepdim=True)
        cand = (shi >= best_lo - HARD_TIE - HARD_ETA) & M
        olo = torch.where(cand[..., None], vlo_j, BIG).amin(-2)
        ohi = torch.where(cand[..., None], vhi_j, -BIG).amax(-2)
        return (olo + ohi) / 2, _infl((olo + ohi) / 2, (ohi - olo) / 2)
    if attn == "softmax":
        Mx = shi.masked_fill(~M, -BIG).amax(-1, keepdim=True)
        Ehi = torch.exp(shi - Mx).masked_fill(~M, 0)
        Elo = torch.exp(slo - Mx).masked_fill(~M, 0)
        Shi, Slo = Ehi.sum(-1, keepdim=True), Elo.sum(-1, keepdim=True)
        wlo = (Elo / (Elo + Shi - Ehi).clamp(min=1e-300)).clamp(0, 1) * (1 - 1e-12)
        whi = (Ehi / (Ehi + Slo - Elo).clamp(min=1e-300)).clamp(0, 1) * (1 + 1e-12)
        wlo, whi = wlo.masked_fill(~M, 0), whi.masked_fill(~M, 0)
        wc, wr = (wlo + whi) / 2, (whi - wlo) / 2
        vcj, vrj = (vlo_j + vhi_j) / 2, (vhi_j - vlo_j) / 2
        oc, orr = mul(wc[..., None], wr[..., None], vcj, vrj)
        oc, orr = oc.sum(-2), orr.sum(-2)
        olo = torch.where(M[..., None], vlo_j, BIG).amin(-2)
        ohi = torch.where(M[..., None], vhi_j, -BIG).amax(-2)
        lo, hi = torch.maximum(oc - orr, olo), torch.minimum(oc + orr, ohi)
        return (lo + hi) / 2, _infl((lo + hi) / 2, (hi - lo) / 2)
    raise ValueError(attn)


def logit_box_tf(P, allowed):
    N, T, V = allowed.shape
    sd = P.sd
    rest = None
    for l in range(P.cfg["layers"]):
        pre = f"blocks.{l}."
        ac, ar = _attn(P, l, rest, allowed, T)
        oc, orr = lin(ac, ar, sd[pre + "o.weight"], sd[pre + "o.bias"])
        rest = (oc, orr) if rest is None else (rest[0] + oc, _infl(rest[0] + oc, rest[1] + orr))
        fc, fr = _res_lin(P, rest, sd[pre + "f1.weight"], sd[pre + "f1.bias"], T)
        fc, fr = relu(fc, fr)
        mc, mr = lin(fc, fr, sd[pre + "f2.weight"], sd[pre + "f2.bias"])
        rest = (rest[0] + mc, _infl(rest[0] + mc, rest[1] + mr))
    return _res_lin(P, rest, sd["out.weight"], sd["out.bias"], T)


def logit_box_rnn(P, allowed):
    N, T, V = allowed.shape
    sd = P.sd
    W, b, U = sd["W.weight"], sd["W.bias"], sd["U.weight"]
    H = W.shape[0]
    hc, hr = torch.zeros(N, H), torch.zeros(N, H)
    outs_c, outs_r = [], []
    for t in range(T):
        pc, pr = lin(hc, hr, W, b)
        c = pc[:, None] + (U[None] if P.TE is None else P.TE[0][:, t])   # per own token [N, V, H]
        c, r = relu(c, _infl(c, pr[:, None].expand_as(c)))
        oc, orr = lin(c, r, sd["out.weight"], sd["out.bias"])
        outs_c.append(oc); outs_r.append(orr)
        hc, hr = hull(c[:, None], r[:, None], allowed[:, t:t + 1])
        hc, hr = hc[:, 0], hr[:, 0]
    return torch.stack(outs_c, 1), torch.stack(outs_r, 1)


def logit_box(P, allowed):
    """Per (position, own token) logit boxes c, r [N, T, V_in, V_out]; rows of disallowed tokens are meaningless."""
    P.TE = None
    c, r = logit_box_rnn(P, allowed) if P.arch == "rnn" else logit_box_tf(P, allowed)
    return c.expand(allowed.shape[0], -1, -1, -1), r.expand(allowed.shape[0], -1, -1, -1)


def ub_log_odds(lo, hi, S):
    """Upper bound on log(P(S) / P(not S)) under softmax, for logits in [lo, hi]; S bool broadcastable to [..., V]."""
    ninf = torch.tensor(float("-inf"))
    a = torch.logsumexp(torch.where(S, hi, ninf), -1)
    b = torch.logsumexp(torch.where(S, ninf, lo), -1)
    return a - b


def logit_box_concrete(P, x):
    """Concrete sequences x [N, T] (model inputs): boxes [N, T, V_out] (radius = rounding allowance, and hard-attention
    near-ties hulled), one own-token slot per position."""
    emb = P.sd["U.weight"] if P.arch == "rnn" else P.sd["tok.weight"]
    xt = torch.as_tensor(x, dtype=torch.long)
    allowed = torch.ones(*xt.shape, 1, dtype=torch.bool)
    P.TE = (emb[xt][:, :, None],)
    try:
        c, r = logit_box_rnn(P, allowed) if P.arch == "rnn" else logit_box_tf(P, allowed)
    finally:
        P.TE = None
    return c[:, :, 0], r[:, :, 0]
