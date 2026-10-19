"""T385 truths (SAMPLING, truth only): central finite differences of P(E1) along the pre-registered directions
(dirs.py), by conditional-law MC with common random numbers (the same draws feed W and every W +- eps d).
Ladder eps in LADDER. Per direction and level: hits+, hits-, both (for the paired variance). The truth of
d log P / d eps is the symmetric log-difference (log H+ - log H-) / (2 eps) at the largest level whose realised
changes satisfy |log(P+/P)| <= .3 and |log(P-/P)| <= .3 (the smallest level if none does); s.e. by the delta method
on the paired indicators. usage: python truth.py model N -> results/truth_<model>.json (checkpointed every 2e6)"""
import json, os, sys, time
import numpy as np
import core, dirs

LADDER = [0.03, 0.01, 0.003, 0.001]
SEED = {"orig": 385100, "dev3_s528": 385101}


def summarize(nm, names, n, pg, h0, hp, hm, hb):
    P = pg * h0 / n; out = dict(model=nm, n=n, p_gap=pg, hits0=int(h0), P=P, P_se=pg * np.sqrt(h0 * (1 - h0 / n)) / n,
                                ladder=LADDER, names=names, rows=[])
    for a, nmd in enumerate(names):
        lv = []
        for l, e in enumerate(LADDER):
            p, m, b = hp[a, l], hm[a, l], hb[a, l]
            if p == 0 or m == 0:
                lv.append(dict(eps=e, hp=int(p), hm=int(m), ok=False, dlog=float("nan"), se=float("nan"))); continue
            # Var(log Hp - log Hm) by the delta method; Var(a)=p(1-p/n), Cov(a,c)=b - p m / n (sums of indicators)
            var = (p * (1 - p / n)) / p ** 2 + (m * (1 - m / n)) / m ** 2 - 2 * (b - p * m / n) / (p * m)
            dl = (np.log(p) - np.log(m)) / (2 * e); se = np.sqrt(max(var, 0)) / (2 * e)
            ok = abs(np.log(p / h0)) <= .3 and abs(np.log(m / h0)) <= .3
            lv.append(dict(eps=e, hp=int(p), hm=int(m), both=int(b), ok=bool(ok), dlog=float(dl), se=float(se),
                           logratio_plus=float(np.log(p / h0)), logratio_minus=float(np.log(m / h0))))
        oks = [l for l in range(len(LADDER)) if lv[l]["ok"]]; pick = oks[0] if oks else len(LADDER) - 1
        out["rows"].append(dict(name=nmd, levels=lv, pick=pick, truth=lv[pick]["dlog"], truth_se=lv[pick]["se"]))
    return out


def main(nm, N, chunk=200_000):
    model = core.load(nm); w0 = core.flat(model); names, D = dirs.all_dirs(nm); law = core.CondLaw(core.CASES[nm])
    var = [[(core.unflat(w0 + e * d), core.unflat(w0 - e * d)) for e in LADDER] for d in D]
    rng = np.random.default_rng(SEED[nm]); path = f"results/truth_{nm}.json"
    shape = (len(D), len(LADDER)); hp = np.zeros(shape, int); hm = np.zeros(shape, int); hb = np.zeros(shape, int)
    n = 0; h0 = 0; t = time.time()
    while n < N:
        x, pos = law.draw(rng, chunk); p3 = pos[:, 2]
        h0 += int((core.logits(model, x).argmax(1) == p3).sum())
        for a in range(len(D)):
            for l in range(len(LADDER)):
                A = core.logits(var[a][l][0], x).argmax(1) == p3; C = core.logits(var[a][l][1], x).argmax(1) == p3
                hp[a, l] += int(A.sum()); hm[a, l] += int(C.sum()); hb[a, l] += int((A & C).sum())
        n += chunk
        if n % 2_000_000 == 0 or n >= N:
            out = summarize(nm, names, n, law.p_gap, h0, hp, hm, hb); out["sec"] = time.time() - t
            out["forward_passes"] = n * (1 + 2 * hp.size); json.dump(out, open(path, "w"), indent=1)
            print(nm, n, "P %.4e" % out["P"], "%.0fs" % out["sec"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(float(sys.argv[2])))
