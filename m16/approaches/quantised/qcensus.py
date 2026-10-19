"""T352 (3): census of reachable hidden states of quantised models (T348-style). N inputs x ~ N(0, I_10) (discretised for
grid variants), forward with the model's inference quantiser (float32: states are discrete; a float32/float64
disagreement needs a pre-activation within ~1e-6 of a threshold and is ignored for a census), state codes
k = round(h / s) in [0, L-1]^16 (binary: h itself). Key: binary 16 bits, int4 64 bits (exact); int8 128 bits mixed to
64 (collision bound checked by a second mixer at the last step). Distinct states per step t = 1..10 by GPU sort +
unique in P hash partitions (all steps per pass). Reports distinct D_t, singletons f1, exp(H).
usage: python qcensus.py CKPT.pt NAME[,NAME..] [N] [P] -> census_<NAME>.json"""
import json, math, os, sys, time
import torch as th
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from qeval import disc  # noqa: E402
DEV = os.environ.get("QDEV", "cuda"); B = 1 << 20 if DEV == "cuda" else 1 << 16
M1, M2 = -7046029254386353131, -4658895280553007687
LV = {1: 1, 2: 15, 3: 255}


def mix(h, w, m):
    h = (h ^ w) * m
    return h ^ (h >> 31)


def keys(k, act, which=1):
    """k (m, H) int64 codes -> (m,) int64: exact for binary H <= 64 and int4 H = 16, else a 64-bit mix of 64-bit words"""
    Hn = k.shape[1]; bits = 1 if act == 1 else (4 if act == 2 else 8); per = 64 // bits
    words = [(k[:, i:i + per] << (bits * th.arange(min(per, Hn - i), device=k.device))).sum(1) for i in range(0, Hn, per)]
    if len(words) == 1:
        return words[0]                                                           # exact (wraps exactly at 64 bits)
    m = M1 if which == 1 else M2; h = th.full_like(words[0], 0x632BE59BD9B4E019)
    for w in words:
        h = mix(h, w, m)
    return h


def run(W, act, s, grid, N, P, seed):
    Wi, Wh = W[0], W[1]; res = {t: dict(coc={}, sl=0.0, n=0) for t in range(1, 11)}
    res["t10_mixer2"] = dict(coc={}, sl=0.0, n=0)
    for part in range(P):
        buf = {t: [] for t in res}; cbuf = {t: [] for t in res}; g = th.Generator(device=DEV); g.manual_seed(seed); done = 0
        while done < N:
            m = min(B, N - done); x = disc(th.randn(m, 10, generator=g, device=DEV), grid)
            h = th.zeros(m, Wh.shape[0], device=DEV)
            for t in range(1, 11):
                u = h @ Wh.T + x[:, t - 1:t] * Wi[None]
                k = (u > 0).long() if act == 1 else th.clamp(th.round(u / s), 0, LV[act]).long()
                h = k.float() if act == 1 else k.float() * s
                kk = keys(k, act); sel = (mix(kk, th.zeros_like(kk), M2) >> 59) & (P - 1)
                u_, c_ = th.unique(kk[sel == part], return_counts=True)       # per-batch dedupe (low-cardinality steps)
                buf[t].append(u_.clone()); cbuf[t].append(c_.clone())     # clone: CUDA unique returns views of m-long buffers
                if t == 10:
                    k2 = keys(k, act, 2) if (act == 3 or (act == 2 and k.shape[1] > 16)) else kk
                    u_, c_ = th.unique(k2[((mix(k2, th.zeros_like(k2), M2) >> 59) & (P - 1)) == part], return_counts=True)
                    buf["t10_mixer2"].append(u_.clone()); cbuf["t10_mixer2"].append(c_.clone())
            done += m
        for t in res:
            K = th.cat(buf[t]); CN = th.cat(cbuf[t]); buf[t] = None; cbuf[t] = None
            if len(K):
                K, o = th.sort(K); CN = CN[o]; uk, inv = th.unique_consecutive(K, return_inverse=True)
                c = th.zeros(len(uk), dtype=CN.dtype, device=K.device).index_add_(0, inv, CN)
                cc, nc = th.unique(c, return_counts=True)
                for a_, b_ in zip(cc.tolist(), nc.tolist()):
                    res[t]["coc"][a_] = res[t]["coc"].get(a_, 0) + b_
                res[t]["sl"] += float((c.double() * c.double().log()).sum()); res[t]["n"] += int(CN.sum())
            del K, CN
        if DEV == "cuda":
            th.cuda.empty_cache()
    out = {}
    for t, r in res.items():
        D = sum(r["coc"].values()); n = r["n"]
        out[str(t)] = dict(distinct=D, singletons=r["coc"].get(1, 0), n=n, exp_entropy=math.exp(math.log(n) - r["sl"] / n) if n else 0)
    return out


if __name__ == "__main__":
    f = sys.argv[1]; names = sys.argv[2].split(","); N = int(float(sys.argv[3])) if len(sys.argv) > 3 else 10 ** 9
    P = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    d = th.load(f, weights_only=False); spec = d["meta"]["spec"]; idx = {sp["name"]: k for k, sp in enumerate(spec)}
    for nm in names:
        k = idx[nm]; sp = spec[k]; t0 = time.time()
        W = [d[a][k].float().to(DEV) for a in ("Wi", "Wh")]
        out = run(W, sp["act"], float(d["s"][k]), sp["grid"], N, P, 3521000 + k)
        json.dump(dict(model=nm, spec=sp, s=float(d["s"][k]), N=N, P=P, seed=3521000 + k, sec=time.time() - t0, steps=out),
                  open(os.path.join(HERE, f"census_{nm}.json"), "w"), indent=1)
        print(nm, {t: v["distinct"] for t, v in out.items()}, "%.0fs" % (time.time() - t0), flush=True)
