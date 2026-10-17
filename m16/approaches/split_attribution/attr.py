"""T195: instrumented recipe108 (frozen, eps .1). Same estimator as estT_fix/estT.py (copied unchanged as estT.py here);
_finish/_margin_split are re-written below line by line with hooks that never touch the estimator's ops/state, plus:
  per resolved split (cell c, gate k at input index t): perm/sigma of the cell's ordering chamber (so every input's
  rank is exact), the gate row a = A[k], the contrast attribution
      con_i = a_i * (mean of pos rays_i - mean of neg rays_i)      (sum_i con_i = a.(mean pos - mean neg) > 0)
  i.e. how much input i's variation across the cell drives the gate's sign change, masses of c and of the minority
  side, and the EXACT class effect within the recipe: for every step-t piece on k's minority side, the error mass of
  the recipe's downstream subtree with gate k true (the normal run) minus with gate k forced to its frozen value
  (counterfactual, separate Ops, no recording). dE = net error change, absdE = sum over pieces of |change|.
  Also the sampled T164 DIAG test (64 Dirichlet points + minority rays; true network vs gate k frozen), labelled sampled.
usage: python attr.py H T seed [cf=1] -> out/H{H}_T{T}_s{seed}.npz (+ resumable per-half checkpoints in data/)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from estT import clip, _ref, _weights_to_margin, _frozen_margin, _adjacent_pairs, _cell_error, TOL, TIE, STATS
from m43 import Ops

REC = []            # split records (dicts)
CF = [False]        # inside a counterfactual subtree: no recording
CUR = {}            # current half: perm, sigma
OPT = dict(cf=True, diag=True, masses=True)
RNG = [np.random.default_rng(1)]


def _record(c, A, g, k, t, model, truth):
    Wi, Wh, Wo = model; T = Wo.shape[0]; st = dict(STATS)
    a = A[k].copy(); s = c.R @ a; sc = TOL * np.linalg.norm(a)
    pos, neg = s > sc, s < -sc
    con = a * (c.R[pos].mean(0) - c.R[neg].mean(0))
    mag = np.abs(a) * np.abs(c.R).mean(0)
    ins, out = clip(c, a, Ops()); mino = out if g[k] > 0 else ins
    r = dict(t=t, k=k, frozen=float(g[k]), perm=CUR["perm"], sigma=CUR["sigma"], truth=truth, a=a, con=con, mag=mag,
             nrays=len(c.R), dE=0.0, absdE=0.0, npieces=0, m_cell=np.nan, m_min=np.nan, s_cls=-1, s_err=-1)
    if OPT["masses"]:
        r["m_cell"] = estT.mass(c, Ops()); r["m_min"] = estT.mass(mino, Ops()) if mino is not None else 0.0
    if OPT["diag"] and mino is not None:
        P = np.vstack([mino.R, RNG[0].dirichlet(np.ones(len(mino.R)), 64) @ mino.R])
        pre = P @ A.T; res = []
        for frz in (False, True):
            G = (pre > 0).astype(float)
            if frz:
                G[:, k] = g[k]
            h = pre * G
            for s_ in range(t + 1, T):
                h = np.maximum(h @ Wh.T + P[:, s_:s_ + 1] * Wi[None], 0)
            res.append((h @ Wo.T).argmax(1))
        r["s_cls"] = int(np.any(res[0] != res[1])); r["s_err"] = int(np.any((res[0] == truth) != (res[1] == truth)))
    STATS.clear(); STATS.update(st)
    REC.append(r)
    return len(REC) - 1


def _margin_split(c, A, model, t, truth, n_true, eps, ops, exact, done=(), tags=()):
    """estT._margin_split verbatim, returning (cell, g, tags): tags = indices of the records whose minority side the
    piece lies on."""
    ref = _ref(c, n_true, ops)
    S = c.R @ A.T; ops.add(len(c.R) * A.shape[0] * (A.shape[1] + 1))
    g = np.zeros(len(A)); cut = []
    for k in range(len(A)):
        sc = TOL * max(np.linalg.norm(A[k]), 1e-300)
        if S[:, k].min() >= -sc:
            g[k] = 1.0 if S[:, k].max() > sc else float(A[k] @ ref > 0)
        elif S[:, k].max() <= sc:
            g[k] = 0.0
        else:
            g[k] = float(A[k] @ ref > 0); ops.add(A.shape[1] + 1)
            if k not in done:
                cut.append(k)
    if not done:
        STATS["cut"] += len(cut)
    if not cut:
        return [(c, g, tags)]
    if exact:
        trig = [cut[0]]
    else:
        w = _weights_to_margin(model, t, truth, ops)
        pairs = _adjacent_pairs(c, ops)
        trig = []
        for k in cut:
            sk = S[:, k]
            mino = (sk > 0) != (g[k] > 0)
            P = [c.R[q] for q in np.flatnonzero(mino)]; sh = [abs(sk[q]) * w[k] for q in np.flatnonzero(mino)]
            for a, b in pairs:
                if (sk[a] > 0) != (sk[b] > 0) and sk[a] != 0 and sk[b] != 0:
                    pp = abs(sk[a]) * c.R[b] + abs(sk[b]) * c.R[a]; P.append(pp / np.linalg.norm(pp)); sh.append(0.0)
                    ops.add(3 * A.shape[1] + 3)
            P = np.array(P); sh = np.array(sh)
            mm = _frozen_margin(model, t, A, g, truth, P, ops)
            score = np.max(sh / (np.abs(mm) + 1e-300))
            flips = mm.min() < 0 < mm.max() and sh.max() >= eps * np.abs(mm).max()
            ops.add(3 * len(P))
            if score >= 1 - TIE or flips:
                trig.append(k)
        if not trig:
            return [(c, g, tags)]
    STATS["resolved"] += len(trig)
    rid = {}
    if not CF[0] and not exact:
        for k in trig:
            rid[k] = _record(c, A, g, k, t, model, truth)
    pieces = [(c, tags)]
    for k in trig:
        nxt = []
        for q, tg in pieces:
            for side, r in zip((1.0, 0.0), clip(q, A[k], ops)):
                if r is not None:
                    nxt.append((r, tg + ((rid[k],) if k in rid and side != g[k] else ())))
        pieces = nxt
    out = []
    for q, tg in pieces:
        out += _margin_split(q, A, model, t, truth, n_true, eps, ops, exact, tuple(done) + tuple(trig), tg)
    return out


def _finish(c, g, A, t, model, truth, n_true, eps, ops, exact):
    """estT._finish (recipe108 branch) verbatim plus the counterfactual per minority piece."""
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]; E = np.eye(T)
    At = Wh @ (g[:, None] * A) + np.outer(Wi, E[t]); ops.add(H * int(g.sum()) * t)
    err = 0.0
    for c2, g2, tg in _margin_split(c, At, model, t, truth, n_true, eps, ops, exact or t in estT.EXACT_STEPS[0]):
        e = _finish(c2, g2, At, t + 1, model, truth, n_true, eps, ops, exact) if t < T - 1 else \
            _cell_error(c2, g2, At, Wo, truth, ops)
        err += e
        if tg and OPT["cf"] and not CF[0]:
            for ri in tg:
                r = REC[ri]; g3 = g2.copy(); g3[r["k"]] = r["frozen"]
                st = dict(STATS); CF[0] = True
                try:
                    e3 = _finish(c2, g3, At, t + 1, model, truth, n_true, eps, Ops(), exact) if t < T - 1 else \
                        _cell_error(c2, g3, At, Wo, truth, Ops())
                finally:
                    CF[0] = False; STATS.clear(); STATS.update(st)
                r["dE"] += e - e3; r["absdE"] += abs(e - e3); r["npieces"] += 1
    return err


def run_half(h, model, eps, ops):
    Wi = model[0]; T = model[2].shape[0]; E = np.eye(T)
    CUR.update(perm=tuple(int(p) for p in h["perm"]), sigma=float(h["sigma"]))
    g1 = (h["sigma"] * Wi > 0).astype(float); ops.add(len(Wi))
    A1 = np.outer(Wi, E[0])
    return _finish(h["cone"], g1, A1, 1, model, h["truth"], h["n_true"], eps, ops, False)


KEYS = ["t", "k", "frozen", "sigma", "truth", "nrays", "dE", "absdE", "npieces", "m_cell", "m_min", "s_cls", "s_err"]


def pack(recs, T):
    o = {key: np.array([r[key] for r in recs], float) for key in KEYS}
    for key in ("perm",):
        o[key] = np.array([r[key] for r in recs], int).reshape(-1, T)
    for key in ("a", "con", "mag"):
        o[key] = np.array([r[key] for r in recs], float).reshape(-1, T)
    return o


def main():
    from zoo44 import zoo
    H, T, seed = (int(v) for v in sys.argv[1:4]); OPT["cf"] = len(sys.argv) < 5 or sys.argv[4] != "0"
    model = zoo(H, T, seed); name = f"H{H}_T{T}_s{seed}"
    ck = os.path.join(HERE, "..", "..", "..", "data", "split_attribution", name); os.makedirs(ck, exist_ok=True)
    estT.MODE[0] = "recipe108"; estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0)
    halves = estT.halves(T); estT._HALVES[T] = halves
    errs = []; nops = 0; t0 = time.time(); allrec = []
    for i, h in enumerate(halves):
        fn = os.path.join(ck, f"half{i:04d}.npz")
        if os.path.exists(fn):
            z = np.load(fn, allow_pickle=True); st = json.loads(str(z["meta"]))
            errs.append(st["err"]); nops += st["ops"]
            for key in ("cells", "cut", "resolved", "simplices"):
                estT.STATS[key] += st[key]
            allrec.append({key: z[key] for key in z.files if key != "meta"}); continue
        REC.clear(); RNG[0] = np.random.default_rng([seed, T, i]); o = Ops(); s0 = dict(estT.STATS)
        e = run_half(h, model, 0.1, o)
        meta = dict(err=e, ops=o.n, **{key: estT.STATS[key] - s0[key] for key in s0})
        p = pack(REC, T) if REC else {key: np.zeros((0, T) if key in ("perm", "a", "con", "mag") else 0) for key in KEYS + ["perm", "a", "con", "mag"]}
        np.savez(fn + ".tmp.npz", meta=json.dumps(meta), **p); os.replace(fn + ".tmp.npz", fn)
        errs.append(e); nops += o.n; allrec.append(p)
        if i % 20 == 0:
            print(f"{name} half {i}/{len(halves)} {time.time() - t0:.0f}s recs {len(REC)}", flush=True)
    err = 0.0
    for e in errs:
        err += e
    rec = {key: np.concatenate([p[key] for p in allrec]) for key in allrec[0]}
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    summ = dict(name=name, acc=1.0 - err, ops=nops, sec=time.time() - t0, cf=OPT["cf"], **estT.STATS)
    np.savez_compressed(os.path.join(HERE, "out", name + ".npz"), meta=json.dumps(summ), **rec)
    print(summ, flush=True)


if __name__ == "__main__":
    main()
