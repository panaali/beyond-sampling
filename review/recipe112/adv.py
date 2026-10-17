"""T113 item 3: adversarial weights for RECIPE112's margin test (it is a per-gate, test-point check, not a cell
certificate).
  part split: function-preserving neuron splitting (neuron k -> n identical copies, each with 1/n of the outgoing
              weights). The truth is unchanged; each copy's shift bound is 1/n of the original, so per-gate tests
              can all pass while the copies switch together.
  part climb: accuracy-preserving hill climb inside the box |dw| <= 0.1 (= 2 sigma) around each clean zoo model,
              maximising |estimate - truth| (3 restarts x 60 steps; deterministic rng).
  part diag:  the worst models from both parts re-run with two diagnostic variants of the test (rec112.P):
              dense (interior grid points of the minority region) and joint (sum of the shift bounds of all frozen
              gates); which one removes the error says which gap caused it.
Estimates: split part uses the reviewed est75 RECIPE112 (rec112 differs on split models by tolerance ties of the
duplicated identical gate rows; the difference is logged); climb and diag use rec112; truth exact.py
(and own exact_mine on the split models).
usage: OMP_NUM_THREADS=1 python adv.py split|climb|diag  -> out/adv_<part>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import exact as EX
import est75 as E75
from m43 import Ops
import mine as M
import rec112 as R

RUNGS = (4, 5, 6, 8)


def est(m, pr=R.P):
    return R.rec112(R.as_lists(m), pr)


def part_split():
    out = []
    for H in RUNGS:
        for zs in range(5):
            base = M.zoo_w(H, zs)
            for j, m in enumerate([base] + M.noisy_abs(base, 4, 0.05, 87)):
                tr = EX.exact_acc(m); e0 = est(m) - tr
                for n in (2, 4, 8):
                    worst = (0.0, None); rdiff = 0.0
                    for k in range(H):
                        ms = M.split_neuron(m, k, n)
                        trs = EX.exact_acc(ms); es = E75.est75(ms, Ops(), **E75.RECIPE112)   # the reviewed code
                        rdiff = max(rdiff, abs(est(ms) - es))
                        if n == 2 and k == 0:
                            assert abs(M.exact_mine(ms) - trs) < 1e-8
                        assert abs(trs - tr) < 1e-8, (trs, tr)
                        if abs(es - tr) > abs(worst[0]):
                            worst = (es - tr, k)
                    out.append(dict(H=H, zoo=zs, model=j, n=n, truth=tr, err_orig=e0, worst_err=worst[0], k=worst[1], reimpl_diff=rdiff))
                    print(f"H={H} zoo {zs} model {j} n={n}: truth {tr:.5f} err orig {100*e0:+.4f}pp, worst split "
                          f"neuron {worst[1]} err {100*worst[0]:+.4f}pp (rec112 vs est75 {rdiff:.0e})", flush=True)
            json.dump(out, open(os.path.join(HERE, "out", "adv_split.json"), "w"), indent=0)


def part_climb():
    out = []
    for H in RUNGS:
        for zs in range(5):
            base = M.zoo_w(H, zs); acc0 = EX.exact_acc(base)
            rng = np.random.default_rng(5000 + 10 * H + zs)
            flat0 = np.concatenate([w.ravel() for w in base]); shapes = [w.shape for w in base]

            def unflat(f):
                i = 0; ws = []
                for sh in shapes:
                    k = int(np.prod(sh)); ws.append(f[i:i + k].reshape(sh)); i += k
                return tuple(ws)

            best_all = (0.0, None, None)
            for r in range(3):
                d = np.zeros_like(flat0); m = unflat(flat0 + d); tr = EX.exact_acc(m); cur = abs(est(m) - tr)
                for step in range(60):
                    d2 = np.clip(d + 0.03 * rng.standard_normal(d.shape), -0.1, 0.1)
                    m2 = unflat(flat0 + d2); tr2 = EX.exact_acc(m2)
                    if tr2 < acc0 - 0.01:
                        continue
                    v = abs(est(m2) - tr2)
                    if v > cur:
                        d, cur, tr = d2, v, tr2
                if cur > best_all[0]:
                    best_all = (cur, d.tolist(), tr)
            m = unflat(flat0 + np.array(best_all[1])); e = est(m) - best_all[2]
            out.append(dict(H=H, zoo=zs, clean_acc=acc0, acc=best_all[2], err=e, delta=best_all[1]))
            print(f"H={H} zoo {zs}: clean acc {acc0:.5f}; worst accuracy-preserving |dw|<=0.1: acc {best_all[2]:.5f} "
                  f"err {100*e:+.4f}pp", flush=True)
            json.dump(out, open(os.path.join(HERE, "out", "adv_climb.json"), "w"))


class Dense(R.P):
    dense = 6


class Joint(R.P):
    joint = True


class Both(R.P):
    dense = 6; joint = True


def part_diag():
    sp = json.load(open(os.path.join(HERE, "out", "adv_split.json")))
    cl = json.load(open(os.path.join(HERE, "out", "adv_climb.json")))
    cases = []
    for H in RUNGS:
        rows = sorted([r for r in sp if r["H"] == H and r["n"] == 2], key=lambda r: -abs(r["worst_err"]))[:2]
        for r in rows:
            base = M.zoo_w(H, r["zoo"]); m = ([base] + M.noisy_abs(base, 4, 0.05, 87))[r["model"]]
            cases.append((f"split n=2 H={H} zoo {r['zoo']} model {r['model']} k {r['k']}", M.split_neuron(m, r["k"], 2)))
        rows = sorted([r for r in cl if r["H"] == H], key=lambda r: -abs(r["err"]))[:2]
        for r in rows:
            base = M.zoo_w(H, r["zoo"]); f = np.concatenate([w.ravel() for w in base]) + np.array(r["delta"])
            i = 0; ws = []
            for w in base:
                k = w.size; ws.append(f[i:i + k].reshape(w.shape)); i += k
            cases.append((f"climb H={H} zoo {r['zoo']}", tuple(ws)))
    out = []
    for name, m in cases:
        tr = EX.exact_acc(m); res = {}
        for pname, pr in (("RECIPE112", R.P), ("dense", Dense), ("joint", Joint), ("dense+joint", Both)):
            st = {}; e = R.rec112(R.as_lists(m), pr, st) - tr; res[pname] = dict(err=e, cells=st["cells"])
        out.append(dict(case=name, truth=tr, **res))
        print(f"{name}: truth {tr:.5f} | " + " | ".join(f"{k} {100*v['err']:+.4f}pp ({v['cells']} cells)" for k, v in res.items()), flush=True)
    json.dump(out, open(os.path.join(HERE, "out", "adv_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    t0 = time.time()
    dict(split=part_split, climb=part_climb, diag=part_diag)[sys.argv[1]]()
    print(f"done {time.time() - t0:.0f}s")
