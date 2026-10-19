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
from load import load
from langs import model_input
import cert, bab

torch.set_num_threads(2)
TAUS = (0.5, 0.9, 0.99)


def codes(y, k):
    c = np.zeros(len(y), np.int64)
    for t in range(k):
        c = c * 4 + y[:, t].astype(np.int64)
    return c


def modsum_ub(P, lang, tau, chunk=20000):
    thr = np.log(tau / (1 - tau)) - bab.DELTA
    ub, flagged, ys = 0.0, [], []
    cert.FLOPS[0] = 0
    for y, w in lang.enumerate_chunks(chunk=chunk):
        T = lang.n_small + 2
        y = y[:, :T]
        x = model_input(y, lang.BOS)
        ann = lang.annotate(y)
        c, r = cert.logit_box_concrete(P, x)
        a = torch.from_numpy(ann["ans_pos"])
        lo, hi = (c - r)[torch.arange(len(y)), a], (c + r)[torch.arange(len(y)), a]
        S = torch.ones(len(y), lang.V_out, dtype=torch.bool)
        S[torch.arange(len(y)), torch.from_numpy(ann["answer"])] = False
        bad = (cert.ub_log_odds(lo, hi, S) >= thr).numpy()
        ub += float(w[bad].sum())
        flagged.append(y[bad].astype(np.int8))
    return ub * (1 + 1e-12), np.concatenate(flagged)


def main():
    K0 = int(sys.argv[1])
    for name in sys.argv[2:]:
        model, lang = load(name)
        P = cert.Params(model)
        meta = json.load(open(os.path.join(HERE, "..", "mini_lm", "models", name + ".json")))
        ref = np.load(os.path.join(HERE, "data", f"ref_{name}.npz"))
        out = dict(name=name, K0=K0, taus={})
        for tau in TAUS:
            t0 = time.time()
            truth = meta["small_exact"][f"B{tau}"]
            rb = ref["y"][ref["max_ill"] > tau]
            if lang.name == "dyck2":
                T = lang.T_small
                res = bab.run_dyck(P, lang, T=T, tau=tau, budget_fpe=1e15, fpe_flops=model.flops(T), chunk=128,
                                   start=K0, keep_closed=True)
                leaf_viol = 0
                for k, pre in res["closed"]:
                    leaf_viol += int(np.isin(codes(rb, k), codes(pre, k)).sum())
                ub, fpe = res["ub"], res["fpe"]
                extra = dict(fail_offset_hist=res["fail_offset_hist"], n_closed=int(sum(len(p) for _, p in res["closed"])))
            else:
                ub, fl = modsum_ub(P, lang, tau)
                fpe = cert.FLOPS[0] / model.flops(lang.n_small + 2)
                T = lang.n_small + 2
                base = lambda yy: (yy[:, :T].astype(np.int64) * 7 ** np.arange(T)).sum(1)
                leaf_viol = int((~np.isin(base(rb), base(fl))).sum())
                extra = dict(n_flagged=len(fl))
            r = dict(truth=truth, ub=ub, violation=bool(ub < truth * (1 - 1e-9)), leaf_violations=leaf_viol,
                     ratio=(ub / truth if truth > 0 else None), fpe=fpe, n_ref_bad=int(len(rb)), sec=time.time() - t0, **extra)
            out["taus"][str(tau)] = r
            print(name, tau, {k: v for k, v in r.items() if k != "fail_offset_hist"}, flush=True)
        json.dump(out, open(os.path.join(HERE, "data", f"sound_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
