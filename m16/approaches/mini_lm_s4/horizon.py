"""Certification horizon (T447): fraction of sampled prefix nodes whose closure test passes for the next j positions,
IBP own-token (cert.py) vs zonotope (zono.py, RNN only). Usage: python horizon.py T n_nodes name [name ...]"""
import os, sys, math, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mini_lm"))
from load import load
from langs import model_input
import cert, bab, zono

torch.set_num_threads(1)
TAUS = (0.5, 0.9, 0.99)


def ibp_offsets(P, T, y, k):
    """Per node: for each tau, the number of positions ahead (1..T-1-k) certified before the first failure."""
    p = y[:, :k].astype(np.int8)
    s, d = bab._stacks(p)
    out = {}
    n = len(d)
    top = np.where(d > 0, s[np.arange(n), np.maximum(d - 1, 0)] if k else 0, 0).astype(np.int64)
    al = bab._allowed_dyck(p, d, top, T, True)
    with torch.no_grad():
        c, r = cert.logit_box(P, al)
    lo, hi = c - r, c + r
    return _offsets(lo[:, k + 1:, :4], hi[:, k + 1:, :4], al[:, k + 1:, :4], d, k, T)


def _offsets(lo_a, hi_a, al_a, d, k, T):
    ninf = torch.tensor(float("-inf"))
    ii = torch.arange(k + 1, T)
    m = (ii - k)[None, :]
    dt = torch.from_numpy(d)[:, None]
    zero_ok = ((dt - m <= 0) & ((dt + m) % 2 == 0))[..., None]
    v_tok = cert.ub_log_odds(lo_a[..., None, :], hi_a[..., None, :], torch.eye(4, dtype=torch.bool))
    v_both = cert.ub_log_odds(lo_a, hi_a, torch.tensor([False, False, True, True]))
    own_open = torch.stack([v_tok[..., 0, 3], v_tok[..., 1, 2]], -1)
    v_one = torch.maximum(v_tok[..., 2:, 2], v_tok[..., 2:, 3])
    own_close = torch.where(zero_ok, torch.maximum(v_one, v_both[..., 2:]), v_one)
    v = torch.where(al_a, torch.cat([own_open, own_close], -1), ninf).amax(-1)   # [n, A]
    res = {}
    for tau in TAUS:
        thr = math.log(tau / (1 - tau)) - bab.DELTA
        ok = (v < thr).numpy()
        res[tau] = np.where(ok.all(1), ok.shape[1], (~ok).argmax(1))
    return res


def zono_offsets(P, model, lang, T, y, k):
    p = y[:, :k]
    s, d = bab._stacks(p.astype(np.int8))
    n = len(d)
    top = np.where(d > 0, s[np.arange(n), np.maximum(d - 1, 0)] if k else 0, 0).astype(np.int64)
    al = bab._allowed_dyck(p.astype(np.int8), d, top, T, True)
    x = model_input(np.concatenate([p, np.zeros((n, 1), np.int64)], 1), lang.BOS)[:, :k + 1]
    with torch.no_grad():
        _, hs = model(torch.from_numpy(x), return_hidden=True)
    h0 = hs[:, k].double()
    rec = []
    zono.rnn_closure(P, h0, al[:, k + 1:, :4], lambda a, lo, hi: rec.append((lo, hi)) or torch.ones(n, dtype=torch.bool))
    lo = torch.stack([r[0] for r in rec], 1); hi = torch.stack([r[1] for r in rec], 1)
    return _offsets(lo, hi, al[:, k + 1:, :4], d, k, T)


def main():
    T, n = int(sys.argv[1]), int(sys.argv[2])
    out = {}
    for name in sys.argv[3:]:
        model, lang = load(name); P = cert.Params(model)
        rng = np.random.default_rng(7)
        row = {}
        for A in (1, 2, 3, 4, 6):
            k = T - 1 - A
            y = lang.sample(n, rng, L=T)
            r = {"ibp": ibp_offsets(P, T, y, k)}
            if P.arch == "rnn":
                r["zono"] = zono_offsets(P, model, lang, T, y, k)
            row[A] = {meth: {str(t): float((v[t] >= A).mean()) for t in TAUS} for meth, v in r.items()}
            row[A]["mean_offset"] = {meth: {str(t): float(v[t].mean()) for t in TAUS} for meth, v in r.items()}
        out[name] = row
        print(name, json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(HERE, "data", f"horizon_T{T}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
