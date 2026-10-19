"""T455: certified input-space BaB for E1(t0 = 2.5) on one triple. T418's D3 search (auto_LiRPA CROWN then alpha-CROWN
on CROWN's survivors; split the widest dimension; best-first) with the rigorous box-mass bound rmass.upper as charge and
pop order. Checkpoints every CKPT CPU-s; --resume continues all counters and the curve.
usage: python cbab.py p1,p2,p3 OUT_PREFIX CPU_SEC [--resume] [--K 256] [--nr 48 --ng 96] [--ckpt 1800]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "late_triple_verify"))
import net    # noqa: E402
import lbab   # noqa: E402
import rmass  # noqa: E402

th.set_num_threads(1)
TAIL = 2 * 0.5 * math.erfc(net.U * net.T0 / math.sqrt(2)) + 8 * 0.5 * math.erfc(net.U * net.T0 / 2)   # outside root box


def widest(lo, hi):
    return th.argmax((hi - lo) + 1e-12 * th.arange(9, 0, -1, dtype=net.dt), 1)


def run(a):
    tri = tuple(int(v) for v in a.tri.split(",")); B = lbab.Bounder(tri)
    rc = lambda lo, hi: rmass.upper(lo, hi, a.nr, a.ng, net.minsq(lo, hi))
    ck = a.out + "_ckpt.pt"
    if a.resume and os.path.exists(ck):
        S = th.load(ck); LO, HI, RC, QM = S["lo"], S["hi"], S["rc"], S["qm"]; st = S["st"]; curve = S["curve"]; res = S["res"]
        cpu0 = st["cpu_s"]
    else:
        lo0, hi0 = net.root(tri); LO, HI = lo0[None].clone(), hi0[None].clone(); RC = rc(LO, HI); QM = net.quad_mass(LO, HI)
        st = dict(evals=0, pruned=0, pr_crown=0, pr_alpha=0, cpu_s=0.0, it=0); curve = []; res = []; cpu0 = 0.0
    rng = np.random.default_rng(455 + len(curve)); t0 = time.process_time(); last_log = -1e9; last_ck = 0.0
    el = lambda: cpu0 + time.process_time() - t0
    while len(LO) and el() < a.sec:
        k = min(a.K, len(LO)); idx = th.topk(RC, k).indices; keep = th.ones(len(LO), dtype=th.bool); keep[idx] = False
        blo, bhi = LO[idx], HI[idx]; LO, HI, RC, QM = LO[keep], HI[keep], RC[keep], QM[keep]
        d = widest(blo, bhi); ar = th.arange(k); mid = (blo[ar, d] + bhi[ar, d]) / 2
        h1 = bhi.clone(); h1[ar, d] = mid; l2 = blo.clone(); l2[ar, d] = mid
        clo, chi = th.cat([blo, l2]), th.cat([h1, bhi]); st["evals"] += len(clo); st["it"] += 1
        lb, _ = B(clo, chi, "CROWN"); pr = (lb > 0).any(1); st["pr_crown"] += int(pr.sum())
        if (~pr).any():
            s = th.nonzero(~pr)[:, 0]; lb2, _ = B(clo[s], chi[s], "alpha-CROWN"); p2 = (lb2 > 0).any(1)
            st["pr_alpha"] += int(p2.sum()); pr[s] = p2
        st["pruned"] += int(pr.sum())
        if pr.any() and sum(len(r[0]) for r in res) < 20000:
            s = th.nonzero(pr)[:, 0]; s = s[th.as_tensor(rng.permutation(len(s))[:16])]; res.append((clo[s], chi[s]))
        sv = ~pr
        LO, HI = th.cat([LO, clo[sv]]), th.cat([HI, chi[sv]])
        RC = th.cat([RC, rc(clo[sv], chi[sv])]); QM = th.cat([QM, net.quad_mass(clo[sv], chi[sv])])
        st["cpu_s"] = el()
        if st["cpu_s"] - last_log > a.log_sec or not len(LO) or st["cpu_s"] >= a.sec:
            last_log = st["cpu_s"]
            row = dict(**{k_: v for k_, v in st.items()}, open=len(LO), cert=float(RC.sum()) + TAIL, quad=float(QM.sum()) + TAIL)
            curve.append(row); print(" ".join("%s %s" % (k_, ("%.4e" % v) if isinstance(v, float) else v) for k_, v in row.items()), flush=True)
        if st["cpu_s"] - last_ck > a.ckpt or not len(LO) or st["cpu_s"] >= a.sec:
            last_ck = st["cpu_s"]
            th.save(dict(lo=LO, hi=HI, rc=RC, qm=QM, st=st, curve=curve, res=res), ck + ".tmp"); os.replace(ck + ".tmp", ck)
            json.dump(dict(tri=tri, K=a.K, nr=a.nr, ng=a.ng, tail=TAIL, curve=curve), open(a.out + ".json", "w"), indent=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tri"); ap.add_argument("out"); ap.add_argument("sec", type=float)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--nr", type=int, default=48); ap.add_argument("--ng", type=int, default=96)
    ap.add_argument("--ckpt", type=float, default=1800); ap.add_argument("--log-sec", type=float, default=300)
    run(ap.parse_args())
