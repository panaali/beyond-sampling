"""Metrics for the mini languages (T433). Works on any weighted set of sequences:
exact enumeration (weights = exact probabilities) or sampling (weights = 1/N)."""
import numpy as np
import torch

from langs import model_input

TAUS = (0.5, 0.9, 0.99)


@torch.no_grad()
def logits_of(model, lang, y, chunk=32768):
    model.eval()
    x = model_input(y, lang.BOS)
    out = []
    for i in range(0, len(x), chunk):
        out.append(model(torch.from_numpy(x[i:i + chunk])).double().numpy())
    return np.concatenate(out)


def per_sequence(model, lang, y):
    """Per-sequence quantities: the hard-part accuracy numerator/denominator, illegal mass sum/count,
    KL sum, and max_t p_illegal (for B_tau)."""
    if lang.name == "modsum":  # causal model: nothing after the answer is scored, so drop those columns
        n_last = (y[:, :lang.N] == lang.EQ).argmax(1)
        n_last = np.where((y[:, :lang.N] != lang.EQ).all(1), lang.N, n_last)
        y = y[:, :int(n_last.max()) + 2]
    ann = lang.annotate(y)
    lg_t = torch.from_numpy(logits_of(model, lang, y))
    lp_t = torch.log_softmax(lg_t, -1)
    lg, lp = lg_t.numpy(), lp_t.numpy()
    legal, law = ann["legal"], ann["law"]
    p_ill = (lp_t.exp() * torch.from_numpy(~legal)).sum(-1).numpy()
    N = len(y)
    if lang.name == "dyck2":
        sc = np.ones_like(p_ill, bool)
        pos = ann["correct"] >= 0
        cc = np.where(pos, ann["correct"], 2)
        wc = np.where(cc == 2, 3, 2)
        lc = np.take_along_axis(lg, cc[..., None], -1)[..., 0]
        lw = np.take_along_axis(lg, wc[..., None], -1)[..., 0]
        acc_num = ((lc > lw) & pos).sum(1).astype(float)
        acc_den = pos.sum(1).astype(float)
    else:
        sc = np.zeros_like(p_ill, bool)
        sc[np.arange(N), ann["ans_pos"]] = True
        am = lg[np.arange(N), ann["ans_pos"]].argmax(-1)
        acc_num = (am == ann["answer"]).astype(float)
        acc_den = np.ones(N)
    law_t = torch.from_numpy(law)
    kl = (torch.xlogy(law_t, law_t) - law_t * lp_t.masked_fill(law_t == 0, 0)).sum(-1).numpy()
    return dict(acc_num=acc_num, acc_den=acc_den,
                ill_sum=(p_ill * sc).sum(1), ill_cnt=sc.sum(1).astype(float),
                kl_sum=(kl * sc).sum(1), max_ill=np.where(sc, p_ill, 0).max(1))


def score_chunks(model, lang, chunks):
    """Weighted metrics over an iterable of (y, w) chunks whose weights sum to 1 (exact law) or are 1/N
    (sampling). Binomial standard errors use the number of sequences."""
    S = dict(acc_num=0.0, acc_den=0.0, ill_sum=0.0, ill_cnt=0.0, kl_sum=0.0)
    B = {t: 0.0 for t in TAUS}
    n = 0
    for y, w in chunks:
        r = per_sequence(model, lang, y)
        w = np.asarray(w, float)
        for k in S:
            S[k] += float((w * r[k]).sum())
        for t in TAUS:
            B[t] += float((w * (r["max_ill"] > t)).sum())
        n += len(y)
    out = dict(acc=S["acc_num"] / S["acc_den"], illegal_mass=S["ill_sum"] / S["ill_cnt"],
               nll_excess=S["kl_sum"] / S["ill_cnt"], n_seq=n)
    for t in TAUS:
        out[f"B{t}"] = B[t]
        out[f"B{t}_se"] = float(np.sqrt(max(B[t] * (1 - B[t]), 0.0) / n))
    return out


def score(model, lang, y, w, chunk=100000):
    return score_chunks(model, lang, ((y[i:i + chunk], w[i:i + chunk]) for i in range(0, len(y), chunk)))


def exact_small(model, lang):
    return score_chunks(model, lang, lang.enumerate_chunks(chunk=100000))


def sampled(model, lang, N, seed, **kw):
    rng = np.random.default_rng(seed)
    y = lang.sample(N, rng, **kw)
    return score(model, lang, y, np.full(N, 1.0 / N))
