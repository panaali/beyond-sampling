"""Branch and bound over token prefixes: a certified upper bound on B_tau = P(max_t p_illegal(x, t) > tau) (T447).

Node = a prefix y_0..y_{k-1} (exact probability under the language law). The node's own new position k (predicting
y_k from BOS, y_<k) is concrete. Each node is
  - charged BAD (full mass) if the bound on the log-odds of the illegal set at position k reaches logit(tau) - DELTA
    (sound: an uncertain node is charged, never dropped);
  - CLOSED (certified good) if, for every later position i in k+1..T-1, an interval bound over every continuation
    (cert.logit_box with abstract positions) keeps the log-odds of every illegal set that position can have below
    logit(tau) - DELTA;
  - otherwise OPEN: it is expanded into its legal children.
UB = mass(BAD) + mass(OPEN) = 1 - mass(CLOSED) at any stopping point. Level-wise, largest mass first in a level.

Illegal sets at an abstract position (dyck2, without the stack top): depth d_i in {d - m, .., d + m} with the parity of
d + m (m = i - k unknown tokens). If d_i = 0 is possible: {), ]}; if d_i > 0 is possible: {)} or {]}.
"""
import math
import time
import numpy as np
import torch

import cert

DELTA = 1e-3


def _allowed_dyck(pre, d, top, T, abstract):
    """Model-input token masks [n, T', 5]: BOS, the prefix, then (if abstract) the legal y_k and all four tokens."""
    n, k = pre.shape
    Tp = T if abstract else k + 1
    al = torch.zeros(n, Tp, 5, dtype=torch.bool)
    al[:, 0, 4] = True
    if k:
        al[:, 1:k + 1].scatter_(2, torch.from_numpy(pre.astype(np.int64))[..., None], True)
    if abstract and k + 1 < T:
        al[:, k + 1, :2] = True
        dd = torch.from_numpy(d > 0)
        al[dd, k + 1, torch.from_numpy(2 + top[d > 0]).long()] = True
        al[:, k + 2:, :4] = True
    return al


def run_dyck(P, lang, T=15, tau=0.9, budget_fpe=1e6, fpe_flops=None, q=0.5, chunk=2048, start=0,
             keep_closed=False, max_frontier=4_000_000):
    """start = K0: the frontier starts as every legal prefix of length K0 (exact enumeration); all concrete
    positions 0..k of a node are checked (charging a node at depth K0 instead of its ancestor at the first bad
    position charges the same total mass)."""
    thr = math.log(tau / (1 - tau)) - DELTA
    fpe_flops = fpe_flops or 1.0
    cert.FLOPS[0] = 0
    pre, mass = lang.enumerate_small(T=start, as_int8=True) if start else (np.zeros((1, 0), np.int8), np.ones(1))
    stk, d = _stacks(pre)
    bad_mass = good_mass = 0.0
    traj, closed_codes, fail_off = [], [], np.zeros(T + 1, np.int64)
    t0 = time.time()
    stopped = None
    for k in range(start, T):
        o = np.argsort(-mass, kind="stable")
        pre, stk, d, mass = pre[o], stk[o], d[o], mass[o]
        nxt = []
        abstract = k + 1 < T
        for i0 in range(0, len(mass), chunk):
            if cert.FLOPS[0] / fpe_flops > budget_fpe:
                stopped = (k, i0)
                break
            sl = slice(i0, i0 + chunk)
            p, s, dd, mm = pre[sl], stk[sl], d[sl], mass[sl]
            n = len(mm)
            top = np.where(dd > 0, s[np.arange(n), np.maximum(dd - 1, 0)] if k else 0, 0).astype(np.int64)
            al = _allowed_dyck(p, dd, top, T, abstract)
            with torch.no_grad():
                c, r = cert.logit_box(P, al)
            lo, hi = c - r, c + r                                   # [n, T', V_in, 4]
            ninf = torch.tensor(float("-inf"))
            # concrete position k: exact illegal set (one allowed own token)
            S = torch.zeros(n, 4, dtype=torch.bool)
            dz = torch.from_numpy(dd == 0)
            S[dz, 2:] = True
            S[~dz, 3 - torch.from_numpy(top[dd > 0])] = True  # wrong closer: top 0 -> ']' (3), top 1 -> ')' (2)
            vk = cert.ub_log_odds(lo[:, k], hi[:, k], S[:, None, :])
            bad = (torch.where(al[:, k], vk, ninf).amax(-1) >= thr).numpy()
            if k > start or k == 0:
                pass
            else:
                Sc = torch.from_numpy(_illegal_sets(p))                       # [n, k, 4] exact, positions 0..k-1
                vc = cert.ub_log_odds(lo[:, :k], hi[:, :k], Sc[:, :, None, :])
                bad |= (torch.where(al[:, :k], vc, ninf).amax(-1) >= thr).any(1).numpy()
            if k + 1 >= T:
                closed = ~bad
            elif not abstract:
                closed = np.zeros(n, bool)
            else:
                ii = torch.arange(k + 1, T)
                m = (ii - k)[None, :]
                dt = torch.from_numpy(dd)[:, None]
                zero_ok = ((dt - m <= 0) & ((dt + m) % 2 == 0))[..., None]   # [n, A, 1]
                lo_a, hi_a = lo[:, k + 1:, :4], hi[:, k + 1:, :4]           # own tokens 0..3
                eye = torch.eye(4, dtype=torch.bool)
                v_tok = cert.ub_log_odds(lo_a[..., None, :], hi_a[..., None, :], eye)   # [n, A, own, {s}]
                v_both = cert.ub_log_odds(lo_a, hi_a, torch.tensor([False, False, True, True]))
                own_open = torch.stack([v_tok[..., 0, 3], v_tok[..., 1, 2]], -1)     # opener c: exactly {3 - c}
                v_one = torch.maximum(v_tok[..., 2:, 2], v_tok[..., 2:, 3])           # closer: top unknown
                own_close = torch.where(zero_ok, torch.maximum(v_one, v_both[..., 2:]), v_one)
                v = torch.cat([own_open, own_close], -1)
                v = torch.where(al[:, k + 1:, :4], v, ninf).amax(-1)                 # [n, A]
                okp = v < thr
                closed = okp.all(1).numpy() & ~bad
                first_fail = torch.where(~okp, ii[None, :] - k, T).amin(1).numpy()
                np.add.at(fail_off, first_fail[~closed & ~bad], 1)
            bad_mass += float(mm[bad].sum())
            good_mass += float(mm[closed].sum())
            if keep_closed and closed.any():
                closed_codes.append((k, p[closed].copy()))
            op = ~bad & ~closed
            if op.any():
                nxt.append(_children(p[op], s[op], dd[op], mm[op], top[op], q))
        rest_mass = float(mass[stopped[1]:].sum()) if stopped else 0.0
        child_mass = sum(float(c[3].sum()) for c in nxt)
        traj.append(dict(level=k, fpe=cert.FLOPS[0] / fpe_flops, bad=bad_mass, good=good_mass,
                         open=rest_mass + child_mass, ub=bad_mass + rest_mass + child_mass,
                         n_next=int(sum(len(c[3]) for c in nxt)), sec=time.time() - t0))
        if stopped or not nxt:
            break
        pre = np.concatenate([c[0] for c in nxt]); stk = np.concatenate([c[1] for c in nxt])
        d = np.concatenate([c[2] for c in nxt]); mass = np.concatenate([c[3] for c in nxt])
        if len(mass) > max_frontier:
            stopped = ("frontier", len(mass))
            traj[-1]["stopped"] = "frontier"
            break
    return dict(traj=traj, ub=traj[-1]["ub"], bad=bad_mass, good=good_mass, fpe=cert.FLOPS[0] / fpe_flops,
                stopped=str(stopped), fail_offset_hist=fail_off.tolist(), closed=closed_codes)


def _stacks(pre):
    n, k = pre.shape
    stk = np.zeros((n, k), np.int8); d = np.zeros(n, np.int64); idx = np.arange(n)
    for t in range(k):
        tok = pre[:, t]
        op = tok < 2
        stk[idx[op], d[op]] = tok[op]
        d = d + np.where(op, 1, -1)
    return stk, d


def _illegal_sets(pre):
    """Exact illegal-token masks [n, k, 4] at positions 0..k-1 of each prefix (position t predicts y_t)."""
    n, k = pre.shape
    S = np.zeros((n, k, 4), bool)
    stk = np.zeros((n, k + 1), np.int8); d = np.zeros(n, np.int64); idx = np.arange(n)
    for t in range(k):
        z = d == 0
        S[z, t, 2:] = True
        top = stk[idx, np.maximum(d - 1, 0)]
        S[idx[~z], t, 3 - top[~z]] = True
        tok = pre[:, t]
        op = tok < 2
        stk[idx[op], d[op]] = tok[op]
        d = d + np.where(op, 1, -1)
    return S


def _children(p, s, d, m, top, q):
    n, k = p.shape
    out = []
    s2 = np.concatenate([s, np.zeros((n, 1), np.int8)], 1)
    for typ in (0, 1):
        ss = s2.copy(); ss[np.arange(n), d] = typ
        out.append((np.concatenate([p, np.full((n, 1), typ, np.int8)], 1), ss, d + 1,
                    m * np.where(d > 0, q / 2, 0.5)))
    c = d > 0
    if c.any():
        out.append((np.concatenate([p[c], (2 + top[c])[:, None].astype(np.int8)], 1), s2[c], d[c] - 1, m[c] * (1 - q)))
    return tuple(np.concatenate([o[j] for o in out]) for j in range(4))
