"""L = 15 soundness check (T447): certified UB vs the exact truth on the 36 baselines, plus a leaf-level check of
every certificate against the harness's own per-sequence max_t p_illegal (ref.py).
  dyck2: bab.run_dyck from start depth K0 (closure tests by cert.py IBP); a violation is UB < truth, or a CLOSED
         prefix with a reference-bad leaf below it.
  modsum: every sequence evaluated by the concrete interval path (no closure is possible without the running sum);
         a violation is a reference-bad sequence that is not charged.
Usage: python soundness.py <K0> name [name ...]  -> data/sound_<name>.json"""
import os, sys, json, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mini_lm"))
from s4load import load_any as load
from langs import model_input
import cert, bab

torch.set_num_threads(2)
TAUS = (0.5, 0.9, 0.99)


def codes(y, k):
    c = np.zeros(len(y), np.int64)
    for t in range(k):
        c = c * 4 + y[:, t].astype(np.int64)
    return c


def modsum_values(P, lang, chunk=20000):
    """Every n <= n_small sequence once: the bound on the illegal log-odds at the answer position (tau-free)."""
    cert.FLOPS[0] = 0
    vs, ws, ys = [], [], []
    T = lang.n_small + 2
    for y, w in lang.enumerate_chunks(chunk=chunk):
        y = y[:, :T]
        x = model_input(y, lang.BOS)
        ann = lang.annotate(y)
        c, r = cert.logit_box_concrete(P, x)
        a = torch.from_numpy(ann["ans_pos"])
        lo, hi = (c - r)[torch.arange(len(y)), a], (c + r)[torch.arange(len(y)), a]
        S = torch.ones(len(y), lang.V_out, dtype=torch.bool)
        S[torch.arange(len(y)), torch.from_numpy(ann["answer"])] = False
        vs.append(cert.ub_log_odds(lo, hi, S).numpy()); ws.append(w); ys.append(y.astype(np.int8))
    return np.concatenate(vs), np.concatenate(ws), np.concatenate(ys)


def main():
    K0 = int(sys.argv[1])
    for name in sys.argv[2:]:
        model, lang = load(name)
        P = cert.Params(model)
        mp = os.path.join(HERE, "..", "mini_lm", "models", name + ".json")
        meta = json.load(open(mp)) if os.path.exists(mp) else None
        ref = np.load(os.path.join(HERE, "data", f"ref_{name}.npz"))
        out = dict(name=name, K0=K0, taus={})
        T = lang.T_small if lang.name == "dyck2" else lang.n_small + 2
        t0 = time.time()
        if lang.name == "dyck2":
            multi, fpe = bab.run_multi(P, lang, T, TAUS, K0, fpe_flops=model.flops(T))
        else:
            mv, mw, my = modsum_values(P, lang)
            fpe = cert.FLOPS[0] / model.flops(T)
        for j, tau in enumerate(TAUS):
            truth = meta["small_exact"][f"B{tau}"] if meta else float(ref[f"B{tau}"])   # S1: ref.py's exact B
            rb = ref["y"][ref["max_ill"] > tau]
            if lang.name == "dyck2":
                res = multi[j]
                leaf_viol = 0
                for k, pre in res["closed"]:
                    leaf_viol += int(np.isin(codes(rb, k), codes(pre, k)).sum())
                ub = res["ub"]
                extra = dict(n_closed=int(sum(len(p) for _, p in res["closed"])), good=res["good"])
            else:
                flag = mv >= np.log(tau / (1 - tau)) - bab.DELTA
                ub, fl = float(mw[flag].sum()) * (1 + 1e-12), my[flag]
                base = lambda yy: (yy[:, :T].astype(np.int64) * 7 ** np.arange(T)).sum(1)
                leaf_viol = int((~np.isin(base(rb), base(fl))).sum())
                extra = dict(n_flagged=len(fl))
            r = dict(truth=truth, ub=ub, violation=bool(ub < truth * (1 - 1e-9)), leaf_violations=leaf_viol,
                     ratio=(ub / truth if truth > 0 else None), fpe=fpe, n_ref_bad=int(len(rb)),
                     sec=time.time() - t0, **extra)
            out["taus"][str(tau)] = r
            print(name, tau, r, flush=True)
        json.dump(out, open(os.path.join(HERE, "data", f"sound_K{K0}_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
