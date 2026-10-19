"""T395: 1e10-input census of error / near-tie gate patterns of M_16,10 (see README pre-registration).
Inputs x ~ N(0, I_10) float64 (CUDA Philox, SEED, chunks of B; every pass regenerates the same stream), gate words
w_t = 16 bits of h_t > 0. Jobs: prefix k = 7..10 on the error and near-tie sets, prefix k = 4..6 on all inputs
(validation), k = 10 error set under a second mixer (collision check). Distinct counting: per batch th.unique (outputs
cloned: CUDA unique returns views of batch-long buffers), merged per hash partition by sort + segment sum; the full
count-of-counts is saved per (set, k). usage: python census10.py N P [seed] -> census10.json"""
import json, math, os, sys, time
import numpy as np, torch as th

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
N = int(float(sys.argv[1])); P = int(sys.argv[2]); SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3950000
DEV = os.environ.get("CDEV", "cuda"); B = 1 << 22 if DEV == "cuda" else 1 << 16
COMPACT = int(float(os.environ.get("COMPACT", 2e8)))   # keys per job buffer before an in-pass merge
if os.environ.get("WEIGHTS"):                                        # Colab: no alg_zoo; weights dumped from m16/common.py
    _w = np.load(os.environ["WEIGHTS"]); Wi0, Wh0, Wo0 = _w["Wi"], _w["Wh"], _w["Wo"]
else:
    from arc_eval import Wi0, Wh0, Wo0  # noqa: E402
Wi, Wh, Wo = (th.as_tensor(w, dtype=th.float64, device=DEV) for w in (Wi0, Wh0, Wo0))
M1, M2 = -7046029254386353131, -4658895280553007687
C1, C2 = 0x632BE59BD9B4E019, 0x2545F4914F6CDD1D
JOBS = [(s, k, 1) for s in ("err", "near") for k in (7, 8, 9, 10)] + [("all", k, 1) for k in (4, 5, 6)] + [("err", 10, 2)]


def mix(h, w, c, m):
    h = (h ^ (w + c)) * m
    return h ^ (h >> 31)


def key(W, k, which):
    if k <= 4:
        h = th.zeros(len(W), dtype=th.int64, device=DEV)
        for t in range(k):
            h = h | (W[:, t] << (16 * t))
        return h
    c, m = (C1, M1) if which == 1 else (C2, M2); h = th.full((len(W),), c, dtype=th.int64, device=DEV)
    for t in range(k):
        h = mix(h, W[:, t] + (t << 16), c, m)
    return h


def part_of(h):
    return ((mix(h, th.zeros_like(h), C2, M2) >> 50) & 1023) % P


def stats(coc, n):
    cs = sorted(coc.items(), reverse=True); D = sum(v for _, v in cs); f1 = coc.get(1, 0); f2 = coc.get(2, 0)
    C = 1 - f1 / n if n else 0.0
    sl = sum(v * c * math.log(c) for c, v in cs); H = math.log(n) - sl / n if n else 0.0
    # Chao-Shen: H = -sum Ca p log(Ca p) / (1 - (1 - Ca p)^n), p = c/n, Ca = coverage (f1 = n guarded)
    Ca = 1 - (f1 if f1 < n else n - 1) / n; Hcs = 0.0
    for c, v in cs:
        pa = Ca * c / n; Hcs -= v * pa * math.log(pa) / (1 - (1 - pa) ** n)
    out = dict(n=n, distinct=D, f1=f1, f2=f2, coverage=C, unseen_mass=1 - C, chao1=D + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2),
               exp_H=math.exp(H), exp_H_chao_shen=math.exp(Hcs), N_delta={}, coc={str(c): v for c, v in cs})
    for dl in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        if dl * n < 10:
            continue
        need = (1 - dl) * n; acc = 0; num = 0
        for c, v in cs:
            if acc + c * v >= need:
                num += math.ceil((need - acc) / c); break
            acc += c * v; num += v
        out["N_delta"][f"{dl:g}"] = dict(N=num, resolved=(1 - C) <= dl / 3)
    return out


def merge(us, cs):
    """merge (key, count) buffers into unique sorted keys with summed counts"""
    K = th.cat(us); CN = th.cat(cs)
    if not len(K):
        return K, CN
    K, o = th.sort(K); CN = CN[o]; uk, inv = th.unique_consecutive(K, return_inverse=True)
    return uk, th.zeros(len(uk), dtype=CN.dtype, device=K.device).index_add_(0, inv, CN)


def batches():
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


res = {f"{s}_k{k}" + ("_mixer2" if w == 2 else ""): dict(coc={}, n=0) for s, k, w in JOBS}; t0 = time.time()
CKPT = os.environ.get("OUT", os.path.join(HERE, "census10.json")) + ".partial"; done_parts = 0
if os.path.exists(CKPT):                                                 # resume after a lost runtime
    ck = json.load(open(CKPT))
    if (ck["N"], ck["P"], ck["seed"]) == (N, P, SEED):
        done_parts = ck["parts_done"]; res = {j: dict(coc={int(a): b for a, b in r["coc"].items()}, n=r["n"]) for j, r in ck["res"].items()}
        print("resumed after part", done_parts, flush=True)
for part in range(done_parts, P):
    ub = {j: [] for j in res}; cb = {j: [] for j in res}; nb = {j: 0 for j in res}
    for W, near, err in batches():
        mask = {"err": err, "near": near, "all": None}
        for s, k, w in JOBS:
            j = f"{s}_k{k}" + ("_mixer2" if w == 2 else ""); hk = key(W if mask[s] is None else W[mask[s]], k, w)
            u, c = th.unique(hk[part_of(hk) == part], return_counts=True); ub[j].append(u.clone()); cb[j].append(c.clone())
            nb[j] += len(u)
            if nb[j] > COMPACT:                                     # in-pass compaction: merge this job's buffers
                K, CN = merge(ub[j], cb[j]); ub[j], cb[j], nb[j] = [K], [CN], len(K)
    for j in res:
        uk, cnt = merge(ub[j], cb[j]); ub[j] = cb[j] = None
        if len(uk):
            cc, nc = th.unique(cnt, return_counts=True)
            for a_, b_ in zip(cc.tolist(), nc.tolist()):
                res[j]["coc"][a_] = res[j]["coc"].get(a_, 0) + b_
            res[j]["n"] += int(cnt.sum())
        del uk, cnt
    if DEV == "cuda":
        th.cuda.empty_cache()
    json.dump(dict(N=N, P=P, seed=SEED, parts_done=part + 1, res=res), open(CKPT + ".tmp", "w")); os.replace(CKPT + ".tmp", CKPT)
    print("part", part + 1, "of", P, "%.0fs" % (time.time() - t0), flush=True)
out = {j: stats(r["coc"], r["n"]) for j, r in res.items()}
json.dump(dict(N=N, P=P, seed=SEED, compact=COMPACT, sec=time.time() - t0, results=out), open(os.environ.get("OUT", os.path.join(HERE, "census10.json")), "w"), indent=1)
for j, v in out.items():
    print(j, "n %d D %d f1 %d C %.4f expH %.0f expHcs %.0f" % (v["n"], v["distinct"], v["f1"], v["coverage"], v["exp_H"], v["exp_H_chao_shen"]), flush=True)
