"""Certified upper bound on B_tau for the dyck2 ReLU RNN by abstract interpretation of the product RNN x stack
automaton (T447). Standard ingredients (interval abstract interpretation of RNNs: Ko et al. 2019 POPQORN, Ryou et al.
2021 Prover, Jacoby et al. 2020 RnnVerify; state merging as in classical abstract interpretation, Cousot & Cousot 1977);
we use plain boxes.

All prefixes of length t that lead to the same GROUP KEY (the exact bracket stack, and optionally the last m tokens)
are merged into one box that contains every hidden state h_t they reach, carrying their total exact probability.
At each step and group: the logits box is checked against the EXACT illegal set of that stack; if the illegal
log-odds can reach logit(tau) - DELTA the whole group mass is charged as bad and the group is dropped (each bad
sequence is charged at or before its first bad time); otherwise every legal next token is applied (interval RNN
step) and the children are merged by key. A push beyond depth DCAP is charged as bad too. UB = sum of charges.
"""
import math
import numpy as np
import torch

import cert

DELTA = 1e-3


def run(P, T=32, tau=0.9, m=0, q=0.5, dcap=24, record=False):
    """m: also key groups by the last m tokens (finer boxes, more groups)."""
    sd = P.sd
    W, b, U = sd["W.weight"], sd["W.bias"], sd["U.weight"]
    Wo, bo = sd["out.weight"], sd["out.bias"]
    Wa, Woa = W.abs(), Wo.abs()
    H = W.shape[0]
    thr = math.log(tau / (1 - tau)) - DELTA
    flops = 0
    # group arrays: depth d, stack bits s, suffix code u, box c/r, mass
    d = np.zeros(1, np.int64); s = np.zeros(1, np.int64); u = np.zeros(1, np.int64)
    c = torch.relu(U[4] + b)[None]; r = torch.zeros(1, H); mass = np.ones(1)
    charged, charged_keys, n_groups = 0.0, [], []
    M = 4 ** m
    for t in range(T):
        lc = c @ Wo.T + bo
        lr = cert._infl(lc, r @ Woa.T)
        flops += 4 * len(mass) * H * 4
        S = torch.zeros(len(mass), 4, dtype=torch.bool)
        dz = d == 0
        top = np.where(dz, 0, (s >> np.maximum(d - 1, 0)) & 1)
        S[torch.from_numpy(dz), 2:] = True
        S[torch.from_numpy(~dz), torch.from_numpy(3 - top[~dz])] = True
        bad = (cert.ub_log_odds(lc - lr, lc + lr, S) >= thr).numpy()
        charged += float(mass[bad].sum())
        if record:
            charged_keys.append(set(zip(d[bad].tolist(), s[bad].tolist(), u[bad].tolist())))
        n_groups.append(len(mass))
        if t == T - 1:
            break
        keep = ~bad
        d, s, u, c, r, mass, top = d[keep], s[keep], u[keep], c[keep], r[keep], mass[keep], top[keep]
        pc = c @ W.T + b
        pr = cert._infl(pc, r @ Wa.T)
        flops += 4 * len(mass) * H * H
        kids = []
        for tok in range(4):
            if tok < 2:
                sel = np.ones(len(mass), bool)
                nd, ns = d + 1, s | (tok << d)
                pm = np.where(d > 0, q / 2, 0.5)
            else:
                sel = (d > 0) & (top == tok - 2)
                nd, ns = d - 1, s & ~(1 << np.maximum(d - 1, 0))
                pm = np.full(len(mass), 1 - q)
            if not sel.any():
                continue
            hc = pc[sel] + U[tok]
            lo, hi = (hc - pr[sel]).clamp(min=0), (hc + pr[sel]).clamp(min=0)
            kids.append((nd[sel], ns[sel], (u[sel] * 4 + tok) % M,
                         lo, hi, mass[sel] * pm[sel]))
        nd = np.concatenate([k[0] for k in kids]); ns = np.concatenate([k[1] for k in kids])
        nu = np.concatenate([k[2] for k in kids]); lo = torch.cat([k[3] for k in kids]); hi = torch.cat([k[4] for k in kids])
        nm = np.concatenate([k[5] for k in kids])
        over = nd > dcap
        charged += float(nm[over].sum())
        nd, ns, nu, lo, hi, nm = nd[~over], ns[~over], nu[~over], lo[torch.from_numpy(~over)], hi[torch.from_numpy(~over)], nm[~over]
        key = np.stack([nd, ns, nu], 1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        G = len(uk)
        it = torch.from_numpy(inv)[:, None].expand(-1, H)
        glo = torch.full((G, H), float("inf")).scatter_reduce(0, it, lo, "amin")
        ghi = torch.full((G, H), float("-inf")).scatter_reduce(0, it, hi, "amax")
        mass = np.zeros(G); np.add.at(mass, inv, nm)
        d, s, u = uk[:, 0], uk[:, 1], uk[:, 2]
        c, r = (glo + ghi) / 2, (ghi - glo) / 2
    return dict(ub=charged, fpe_flops=flops, n_groups=n_groups, charged_keys=charged_keys)
