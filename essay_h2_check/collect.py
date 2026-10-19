"""T383 collection (own code). Qwen3-0.6B fp32, MLP hidden activations (input to down_proj, silu(gate)*up).

Corpus windows are defined as in E1 (pile-10k, 64-token windows, at most 4 per document, 18,750 windows);
positions 0-3 of every window are excluded from all statistics.

usage:
  python collect.py stats   pass 1 -> data/t383/stats.npz: per-neuron positive/negative max over the FULL
                            corpus, positive-part sums and log-binned mass histograms (approximate texture)
  python collect.py main    pass 2 -> data/t383/L{LL}.f16 (U, N_WIN*CTX) raw values of the stored units, and
                            data/t383/winmax.npz: for every candidate control direction and window, the max
                            pole value, its position, and the top-1 / top-5 neuron contribution shares there
Both passes are resumable.  Env T383_NWIN limits the number of windows (smoke tests only, into T383_OUT).
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
CTX, PER_DOC, BS, SKIP = 64, 4, 16, 4
N_WIN = int(os.environ.get("T383_NWIN", 18750))
OUT = os.environ.get("T383_OUT", "data/t383")
PARQUET = "data/pile10k/data/train-00000-of-00001-4746b8785c874cc7.parquet"
NL, DM = 28, 3072
N_PLAIN, N_POOL, N_DIR, N_SEL = 11, 64, 19, 8  # dirs 0-7 select, 8-18 are items
POOL_RANGE, MIN_ABOVE = (0.10, 0.60), 100  # amendment 1: pool targets, min tokens above 20% of the pole max
CONSTR = ["Mf", "Rf", "Rp"]  # pre-registered order (README)
EDGES = np.geomspace(1e-4, 1e4, 161)  # 160 log bins for the texture histograms
NB = len(EDGES) - 1

torch.set_num_threads(int(os.environ.get("T383_THREADS", "4")))
os.makedirs(OUT, exist_ok=True)


def windows(tok):
    texts = pq.read_table(PARQUET)["text"].to_pylist()
    wins, doc = [], []
    for d, t in enumerate(texts):
        ids = tok(t[:20000])["input_ids"]
        for k in range(min(PER_DOC, len(ids) // CTX)):
            wins.append(ids[k * CTX:(k + 1) * CTX])
            doc.append(d)
        if len(wins) >= N_WIN:
            break
    return np.array(wins[:N_WIN], dtype=np.int32), np.array(doc[:N_WIN], dtype=np.int32)


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if not os.path.exists(f"{OUT}/tokens.npy"):
        w, d = windows(tok)
        np.save(f"{OUT}/tokens.npy", w)
        np.save(f"{OUT}/doc_of.npy", d)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").eval()
    acts = {}
    for i, layer in enumerate(model.layers):
        layer.mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: acts.__setitem__(i, inp[0].detach()))
    return model, np.load(f"{OUT}/tokens.npy"), acts


def stats():
    model, tokens, acts = load_model()
    ck = f"{OUT}/stats_ck.npz"
    if os.path.exists(ck):
        z = np.load(ck)
        st = {k: z[k] for k in z.files}
    else:
        st = {"done": np.array(0), "mpos": np.zeros((NL, DM)), "mneg": np.zeros((NL, DM)),
              "spos": np.zeros((NL, DM)), "sneg": np.zeros((NL, DM)),
              "hpos": np.zeros((NL, DM, NB)), "hneg": np.zeros((NL, DM, NB)),
              "cpos": np.zeros((NL, DM, NB)), "cneg": np.zeros((NL, DM, NB))}
    edges = torch.tensor(EDGES[1:-1], dtype=torch.float32)
    off = (torch.arange(DM) * NB)[None, :]
    start = int(st["done"])
    for b in range(start, len(tokens), BS):
        with torch.inference_mode():
            model(torch.tensor(tokens[b:b + BS], dtype=torch.long))
            for L in range(NL):
                h = acts[L][:, SKIP:].reshape(-1, DM)
                for v, mk, sk, hk, ck_ in ((h.clamp(min=0), "mpos", "spos", "hpos", "cpos"),
                                           ((-h).clamp(min=0), "mneg", "sneg", "hneg", "cneg")):
                    st[mk][L] = np.maximum(st[mk][L], v.max(0).values.numpy())
                    st[sk][L] += v.sum(0, dtype=torch.float64).numpy()
                    idx = (torch.bucketize(v, edges) + off).reshape(-1)
                    st[hk][L] += torch.bincount(idx, weights=v.reshape(-1).double(), minlength=DM * NB
                                                ).reshape(DM, NB).numpy()
                    st[ck_][L] += torch.bincount(idx, minlength=DM * NB).reshape(DM, NB).numpy()
        end = min(b + BS, len(tokens))
        if (b // BS) % 100 == 99 or end == len(tokens):
            st["done"] = np.array(end)
            np.savez(ck + ".tmp.npz", **st)
            os.replace(ck + ".tmp.npz", ck)
            print(f"stats {end}/{len(tokens)}", flush=True)
    np.savez(f"{OUT}/stats.npz", **st)


def approx_texture(st):
    """Approximate fraction of pole mass above 20% of the pole max, and approximate number of tokens above it,
    from the histograms. Two arrays of shape (2, NL, DM); index 0 = positive pole, 1 = negative pole."""
    out, cnt = np.zeros((2, NL, DM)), np.zeros((2, NL, DM))
    lo, hi = EDGES[:-1], EDGES[1:]
    for si, (mk, sk, hk, ck_) in enumerate((("mpos", "spos", "hpos", "cpos"), ("mneg", "sneg", "hneg", "cneg"))):
        th = 0.2 * st[mk][..., None]
        frac = np.clip((hi - th) / (hi - lo), 0, 1)  # share of each bin above the threshold (linear approx)
        out[si] = (st[hk] * frac).sum(-1) / np.maximum(st[sk], 1e-12)
        cnt[si] = (st[ck_] * frac).sum(-1)
    return out, cnt


def units():
    """Pre-registered unit draw (README). Returns per-layer lists of (kind, neuron_or_-1, sign, extra)."""
    st = dict(np.load(f"{OUT}/stats.npz"))
    tex, cnt = approx_texture(st)
    mpos, mneg = st["mpos"], st["mneg"]
    amax = np.maximum(mpos, mneg)
    alive = amax > 1e-6
    rng = np.random.default_rng(383)
    plain = [(rng.choice(np.flatnonzero(alive[L]), N_PLAIN, replace=False), rng.choice([1, -1], N_PLAIN))
             for L in range(NL)]
    pool = []  # amendment 1: eligible poles only, spread over a fixed texture range (README)
    for L in range(NL):
        cand = np.setdiff1d(np.flatnonzero(alive[L]), plain[L][0])
        ok = cnt[:, L, cand] >= MIN_ABOVE
        t2 = np.where(ok, tex[:, L, cand], -np.inf)
        best = np.where(t2[0] >= t2[1], 1, -1)  # one pole per neuron: the higher-texture eligible one
        t = np.where(best == 1, t2[0], t2[1])
        free = np.isfinite(t)
        sel = []
        for target in np.linspace(*POOL_RANGE, N_POOL):
            k = np.flatnonzero(free)[np.argmin(np.abs(t[free] - target))]
            sel.append(k)
            free[k] = False
        sel = np.array(sel)
        pool.append((cand[sel], best[sel], t[sel]))
    dirs, dsign = np.zeros((NL, DM, len(CONSTR) * N_DIR), np.float32), np.zeros((NL, len(CONSTR) * N_DIR), int)
    for ci, c in enumerate(CONSTR):
        r = np.random.default_rng([383, ci])
        s = r.choice([1, -1], (NL, N_DIR))
        if c == "Mf":
            w = r.standard_normal((NL, DM, N_DIR)) / np.where(alive, amax, np.inf)[..., None]
        elif c == "Rf":
            w = r.choice([1.0, -1.0], (NL, DM, N_DIR)) / np.where(alive, amax, np.inf)[..., None]
        else:  # Rp: pole-aware peak cap, max_t s*w_i*a_i(t) = 1 for every neuron i
            rr = r.choice([1.0, -1.0], (NL, DM, N_DIR))
            peak = np.where(rr * s[:, None, :] > 0, mpos[..., None], mneg[..., None])
            w = rr / np.where(peak > 1e-6, peak, np.inf)
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        dirs[:, :, ci * N_DIR:(ci + 1) * N_DIR] = w
        dsign[:, ci * N_DIR:(ci + 1) * N_DIR] = s
    return plain, pool, dirs, dsign, tex


def main():
    plain, pool, dirs, dsign, _ = units()
    model, tokens, acts = load_model()
    T = len(tokens) * CTX
    nidx = [np.concatenate([plain[L][0], pool[L][0]]) for L in range(NL)]
    U = len(nidx[0]) + dirs.shape[2]
    meta = {"U": U, "T": T, "CTX": CTX, "N_WIN": len(tokens), "constructions": CONSTR, "n_dir": N_DIR, "n_sel": N_SEL,
            "layers": [{"plain": plain[L][0].tolist(), "plain_sign": plain[L][1].tolist(),
                        "pool": pool[L][0].tolist(), "pool_sign": pool[L][1].tolist(),
                        "pool_approx_texture": pool[L][2].tolist(), "dir_sign": dsign[L].tolist()}
                       for L in range(NL)]}
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    np.save(f"{OUT}/dirs.npy", dirs)
    for L in range(NL):
        fn = f"{OUT}/L{L:02d}.f16"
        if not os.path.exists(fn) or os.path.getsize(fn) != U * T * 2:
            with open(fn, "wb") as f:
                f.truncate(U * T * 2)
    nd = dirs.shape[2]
    wm = f"{OUT}/winmax.npz"
    if os.path.exists(wm):
        z = np.load(wm)
        rec = {k: z[k] for k in z.files}
    else:
        rec = {"done": np.array(0), "val": np.zeros((len(tokens), NL, nd), np.float32),
               "pos": np.zeros((len(tokens), NL, nd), np.int16),
               "top1": np.zeros((len(tokens), NL, nd), np.float32), "top5": np.zeros((len(tokens), NL, nd), np.float32)}
    D = torch.tensor(dirs)
    S = torch.tensor(dsign, dtype=torch.float32)
    I = [torch.tensor(x) for x in nidx]
    start = int(rec["done"])
    chunk, c0 = [], start

    def flush(end):
        a = np.concatenate(chunk, 1)  # (NL, rows, U)
        for L in range(NL):
            with open(f"{OUT}/L{L:02d}.f16", "r+b") as f:
                for u in range(U):
                    f.seek((u * T + c0 * CTX) * 2)
                    f.write(np.ascontiguousarray(a[L, :, u]).tobytes())
        rec["done"] = np.array(end)
        np.savez(wm + ".tmp.npz", **rec)
        os.replace(wm + ".tmp.npz", wm)
        print(f"main {end}/{len(tokens)}", flush=True)

    for b in range(start, len(tokens), BS):
        nb = min(BS, len(tokens) - b)
        with torch.inference_mode():
            model(torch.tensor(tokens[b:b + nb], dtype=torch.long))
            rows = []
            for L in range(NL):
                h = acts[L]  # (nb, CTX, DM)
                proj = h @ D[L]  # (nb, CTX, nd)
                rows.append(torch.cat([h[..., I[L]], proj], -1).reshape(nb * CTX, U).numpy().astype(np.float16))
                v = proj[:, SKIP:] * S[L]
                vmax, p = v.max(1)  # (nb, nd)
                hs = h[torch.arange(nb)[:, None], p + SKIP]  # (nb, nd, DM)
                c = hs * D[L].T[None] * S[L][None, :, None]
                rec["val"][b:b + nb, L] = vmax.numpy()
                rec["pos"][b:b + nb, L] = (p + SKIP).numpy()
                rec["top1"][b:b + nb, L] = (c.max(-1).values / vmax).numpy()
                rec["top5"][b:b + nb, L] = (c.topk(5, -1).values.sum(-1) / vmax).numpy()
        chunk.append(np.stack(rows))
        end = b + nb
        if len(chunk) == 50 or end == len(tokens):
            flush(end)
            chunk, c0 = [], end


if __name__ == "__main__":
    {"stats": stats, "main": main}[sys.argv[1]]()
