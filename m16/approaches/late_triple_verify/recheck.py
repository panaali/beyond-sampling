"""T418 E1: re-check T414's CROWN with auto_LiRPA on the root box and on boxes from a re-run of T414's own search,
then looseness / pruning of C, A, AF, ABC-in, ABC-relu on its 200 top-charge open boxes.
usage: python recheck.py T414_BOXES.pt OUT.json [parts: match,loose,abcin,abcrelu] [n_top]"""
import json
import sys
import time

import numpy as np
import torch as th

import net
import lbab

th.set_num_threads(1)
tri = (3, 7, 8); p3 = 8
Bx = th.load(sys.argv[1]); out = sys.argv[2]
parts = sys.argv[3].split(",") if len(sys.argv) > 3 else ["match", "loose"]
ntop = int(sys.argv[4]) if len(sys.argv) > 4 else 200
R = json.load(open(out)) if parts != ["match", "loose"] and __import__("os").path.exists(out) else {}
rng = np.random.default_rng(4180)
C = lbab.Bounder(tri)

if "match" in parts:
    import ibab                                                                    # T414's code (re-check target)
    P, q, lo0, hi0 = ibab.coords(tri)
    A, c = net.ymap(tri); assert th.equal(A, P) and th.equal(c, q)
    ol, oh = Bx["open_lo"], Bx["open_hi"]; sel = th.as_tensor(rng.choice(len(ol), min(2000, len(ol)), replace=False))
    sets = dict(root=(lo0[None], hi0[None]), open=(ol[sel], oh[sel]), pruned=(Bx["pruned_lo"], Bx["pruned_hi"]))
    R["match"] = {}
    for k, (lo, hi) in sets.items():
        lbs = []; lbt = []
        for i in range(0, len(lo), 500):
            lbs.append(C(lo[i:i + 500], hi[i:i + 500], "CROWN")[0]); lbt.append(ibab.margins(lo[i:i + 500], hi[i:i + 500], P, q, p3)[0])
        a_, b_ = th.cat(lbs), th.cat(lbt); rel = ((a_ - b_).abs() / (1 + b_.abs())).max()
        R["match"][k] = dict(n=len(lo), max_abs=float((a_ - b_).abs().max()), max_rel=float(rel),
                             pruned_lirpa=int((a_ > 0).any(1).sum()), pruned_t414=int((b_ > 0).any(1).sum()), PASS=bool(rel <= 1e-6))
        print("match", k, R["match"][k], flush=True)

ch = Bx["open_ch"]; top = th.topk(ch, min(ntop, len(ch))).indices; lo, hi = Bx["open_lo"][top], Bx["open_hi"][top]
M = net.UNet(tri)
with th.no_grad():
    smin = th.stack([M(lo[i] + (hi[i] - lo[i]) * th.as_tensor(rng.uniform(size=(2000, 9)))).min(0).values for i in range(len(lo))])
    sev = th.stack([(M(lo[i] + (hi[i] - lo[i]) * th.as_tensor(rng.uniform(size=(2000, 9)))) <= 0).all(1).any() for i in range(len(lo))])


def loose(lb):
    j = lb.argmax(1); g = (smin[th.arange(len(lb)), j] - lb[th.arange(len(lb)), j]).numpy()
    return dict(median=float(np.median(g)), p10=float(np.quantile(g, .1)), p90=float(np.quantile(g, .9)),
                pruned=int((lb > 0).any(1).sum()), best_lb_median=float(lb.max(1).values.median()))


if "loose" in parts:
    R["top"] = dict(n=len(lo), sampled_prunable=int((smin > 0).any(1).sum()), sampled_event_boxes=int(sev.sum()))
    print("top", R["top"], flush=True)
    for name, meth, opts in [("C", "CROWN", None), ("A", "alpha-CROWN", None), ("AF", "alpha-CROWN", {"fix_interm_bounds": False})]:
        b = C if opts is None else lbab.Bounder(tri)
        if opts:
            b.bm.set_bound_opts({"optimize_bound_args": {"iteration": 20, **opts}})
        t = time.process_time(); lbs = []
        try:
            for i in range(0, len(lo), 50):
                lbs.append(b(lo[i:i + 50], hi[i:i + 50], meth)[0])
            R[name] = dict(**loose(th.cat(lbs)), cpu_s=time.process_time() - t)
        except Exception as e:                                                   # recorded, not hidden
            R[name] = dict(error=repr(e)[:300])
        print(name, R[name], flush=True)
    json.dump(R, open(out, "w"), indent=1)

lbC = C(lo, hi, "CROWN")[0]
for part, mode in [("abcin", "input"), ("abcrelu", "relu")]:
    if part not in parts:
        continue
    abc = lbab.ABC(tri, 10.0, mode); th.set_default_dtype(th.float64); ok = 0; dom = []; st = {}; t = time.process_time()
    for i in range(len(lo)):
        v, d, s = abc(lo[i], hi[i], th.argsort(lbC[i], descending=True)); ok += v; dom.append(d); st[s] = st.get(s, 0) + 1
    R[part] = dict(n=len(lo), verified=ok, statuses=st, domains_mean=float(np.mean(dom)), cpu_s=time.process_time() - t)
    print(part, R[part], flush=True)
    json.dump(R, open(out, "w"), indent=1)
json.dump(R, open(out, "w"), indent=1)
