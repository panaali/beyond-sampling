"""T348: gate-pattern census of M_16,10 for k = 4..10 (diagnostic sampling, not an estimator).
x ~ N(0, I_10) float64 (CUDA Philox, seed SEED, chunks of 2^24; every pass regenerates the same stream), h_t =
relu(Wh h_{t-1} + Wi x_t); gate word w_t = 16 bits of (h_t > 0). Pattern of length k = (w_1..w_k), hashed to 64 bits by
an order-dependent mixer (two independent mixers h1, h2; distinct counts under both bound collisions; exact for k <= 4
since 4 x 16 bits fit). Sets: all inputs; near-tie (min(g12, g23) < .1, top-3 gaps of x); error (argmax(Wo h_10) != p2).
Per (k, set): distinct patterns D (lower bound on N_k), N_delta = patterns with the largest empirical masses covering
1 - delta of the set's sample mass, exp(H) of the empirical pattern distribution, singletons f1 (Good-Turing missing
mass ~ f1 / n). Distinct counting by sort + unique on GPU, every set split into P passes by the top hash bits so a
pass holds ~n / P keys per set; per pass only count-of-counts and sum c log c are kept (host memory small).
Also backward (suffix) patterns, steps 11-k..10, for k = 7..10 (all and error inputs; T346 extension).
usage: python census.py N [P] [seed] -> census.json"""
import json, math, os, sys, time
import numpy as np, torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(float(sys.argv[1])); P = int(sys.argv[2]) if len(sys.argv) > 2 else 4; SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3480000
DEV = os.environ.get("CENSUS_DEV", "cuda"); B = 1 << 24 if DEV == "cuda" else 1 << 18
z = np.load(os.path.join(HERE, "weights.npz"))
Wi, Wh, Wo = (th.as_tensor(z[a], dtype=th.float64, device=DEV) for a in ("Wi", "Wh", "Wo"))
M1, M2 = -7046029254386353131, -4658895280553007687          # 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9 as int64
C1, C2 = 0x632BE59BD9B4E019, 0x2545F4914F6CDD1D


def mix(h, w, c, m):
    h = (h ^ (w + c)) * m
    return h ^ (h >> 31)


def batches():
    """yields per batch: gate words (m, 10) int64, near-tie mask, error mask"""
    g = th.Generator(device=DEV); g.manual_seed(SEED); done = 0
    while done < N:
        m = min(B, N - done); x = th.randn(m, 10, generator=g, device=DEV, dtype=th.float64)
        h = th.zeros(m, 16, dtype=th.float64, device=DEV); W = []
        for t in range(10):
            h = th.relu(h @ Wh.T + x[:, t:t + 1] * Wi[None])
            W.append(((h > 0).long() << th.arange(16, device=DEV)).sum(1))
        v, p = x.topk(3, 1); near = th.minimum(v[:, 0] - v[:, 1], v[:, 1] - v[:, 2]) < 0.1
        err = (h @ Wo.T).argmax(1) != p[:, 1]
        yield th.stack(W, 1), near, err; done += m


def keys(W, k, which, suffix=False):
    """prefix pattern (steps 1..k) or, with suffix, backward pattern (steps 11-k..10)"""
    ts = range(10 - k, 10) if suffix else range(k)
    if k <= 4 and not suffix:                                   # exact: 16 k bits
        h = th.zeros(len(W), dtype=th.int64, device=DEV)
        for t in ts:
            h = h | (W[:, t] << (16 * t))
        return h
    c, m = (C1, M1) if which == 1 else (C2, M2)
    h = th.full((len(W),), c, dtype=th.int64, device=DEV)
    for t in ts:
        h = mix(h, W[:, t] + (t << 16), c, m)
    return h


def part_of(h):
    """partition index from a re-mixed key (raw exact keys have very skewed top bits)"""
    return ((mix(h, th.zeros_like(h), C2, M2) >> 58) & 63) % P


def stats(cnt_of_cnt, slogc, n):
    """cnt_of_cnt: {count c: number of patterns with count c}; slogc = sum c log c"""
    cs = sorted(cnt_of_cnt.items(), reverse=True); D = sum(v for _, v in cs)
    out = dict(n=n, distinct=D, singletons=cnt_of_cnt.get(1, 0), good_turing_missing=cnt_of_cnt.get(1, 0) / max(n, 1),
               exp_entropy=math.exp(math.log(n) - slogc / n) if n else 0.0, N_delta={})
    for dl in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        need = (1 - dl) * n; acc = 0; num = 0
        if dl * n < 1:
            out["N_delta"][f"{dl:g}"] = None; continue          # not supported by the sample
        for c, v in cs:
            if acc + c * v >= need:
                num += math.ceil((need - acc) / c); break
            acc += c * v; num += v
        out["N_delta"][f"{dl:g}"] = num
    return out


def tally(K, coc, sl):
    K, _ = th.sort(K); u, c = th.unique_consecutive(K, return_counts=True)
    cc, nc = th.unique(c, return_counts=True)
    for a, b in zip(cc.tolist(), nc.tolist()):
        coc[a] = coc.get(a, 0) + b
    cf = c.double(); return sl + float((cf * cf.log()).sum())


res = {}; t0 = time.time()
JOBS = []                                                       # (label, k, mixer, suffix, sets)
for k in range(4, 11):
    JOBS.append((f"prefix_k{k}", k, 1, False, ("all", "near", "err")))
    if k == 10:
        JOBS.append((f"prefix_k{k}_mixer2", k, 2, False, ("all",)))
    if k >= 7:
        JOBS.append((f"suffix_k{k}", k, 1, True, ("all", "err")))
for label, k, which, suf, SETS in JOBS:
    coc = {s: {} for s in SETS}; sl = {s: 0.0 for s in SETS}; ns = {s: 0 for s in SETS}
    for part in range(P):
        buf = {s: [] for s in SETS}
        for W, near, err in batches():
            h = keys(W, k, which, suf); sel = part_of(h) == part
            for s in SETS:
                buf[s].append(h[sel] if s == "all" else h[sel & (near if s == "near" else err)])
        for s in SETS:
            K = th.cat(buf[s]); buf[s] = None; ns[s] += len(K)
            if len(K):
                sl[s] = tally(K, coc[s], sl[s])
            del K
        if DEV == "cuda":
            th.cuda.empty_cache()
    res[label] = {s: stats(coc[s], sl[s], ns[s]) for s in SETS}
    print(label, json.dumps({s: dict(D=v["distinct"], n=v["n"], f1=v["singletons"], expH=round(v["exp_entropy"]), Nd=v["N_delta"])
                             for s, v in res[label].items()}), "%.0fs" % (time.time() - t0), flush=True)
    json.dump(dict(N=N, seed=SEED, P=P, results=res), open(os.path.join(HERE, "census.json"), "w"), indent=1)
