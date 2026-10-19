"""T418: input-space BaB over homogeneity-reduced u-boxes with LIBRARY bounds (auto_LiRPA CROWN -> alpha-CROWN, and
optionally alpha-beta-CROWN as a leaf verifier). Same search as T414 (best-first by radial charge, bisection along the
dim maximising |c_d| width_d of the best spec's linear lower bound), different code.
usage: python lbab.py p1,p2,p3 OUT.json CPU_SEC [--stages crown,alpha[,abc]] [--K 256] [--resume OPEN.pt]
       [--abc-timeout 2] [--abc-mode mixed|input|relu]"""
import argparse
import json
import math
import time

import numpy as np
import torch as th

import net

th.set_num_threads(1)
TAIL = 2 * 0.5 * math.erfc(net.U * net.T0 / math.sqrt(2)) + 8 * 0.5 * math.erfc(net.U * net.T0 / 2)


class Bounder:
    def __init__(self, tri, alpha_iter=20):
        from auto_LiRPA import BoundedModule
        self.m = net.UNet(tri).eval()
        self.bm = BoundedModule(self.m, th.zeros(1, 9, dtype=net.dt), device="cpu",
                                bound_opts={"verbosity": 0, "optimize_bound_args": {"iteration": alpha_iter}})
        self.need = {self.bm.output_name[0]: [self.bm.input_name[0]]}

    def __call__(self, lo, hi, method):
        """lower bounds of the 9 margins (N, 9) and the u-coefficients of their linear lower bounds (N, 9, 9)"""
        from auto_LiRPA import BoundedTensor, PerturbationLpNorm
        x = BoundedTensor((lo + hi) / 2, PerturbationLpNorm(x_L=lo, x_U=hi))
        lb, _, A = self.bm.compute_bounds(x=(x,), method=method, bound_upper=False, return_A=True, needed_A_dict=self.need)
        lA = A[self.bm.output_name[0]][self.bm.input_name[0]]["lA"]
        return lb.detach(), lA.detach().reshape(len(lo), 9, 9)


SPLIT = "margin"


def splitdim(lb, cu, lo, hi):
    if SPLIT == "widest":                                         # deviation D3: widest dim, lowest index on ties
        return th.argmax((hi - lo) + 1e-12 * th.arange(9, 0, -1, dtype=net.dt), 1)
    c = cu[th.arange(len(lb)), lb.argmax(1)]
    return (c.abs() * (hi - lo)).argmax(1)


class ABC:
    """alpha-beta-CROWN verifier for 'some margin > 0' on a u-box"""
    def __init__(self, tri, timeout, mode):
        th.set_default_dtype(th.float64)
        from api import ABCrownSolver, ConfigBuilder, input_vars, output_vars
        from api import IOConstraints
        self.IO = IOConstraints; self.xv = input_vars(9); self.yv = output_vars(9)
        cfg = (ConfigBuilder.from_defaults().set("general/device", "cpu").set("general/double_fp", True)
               .set("bab/timeout", timeout).set("attack/pgd_order", "skip").set("general/verbose", 0))
        if mode == "relu":
            cfg.set("bab/branching/input_split/input_dim_threshold", 0)
        elif mode == "mixed":
            cfg.set("bab/branching/branching_input_and_activation", True)
        self.mode = mode
        self.solver = ABCrownSolver(net.UNet(tri).eval(), self.xv, self.yv, config=cfg)
        oc = self.yv[0] > 0
        for j in range(1, 9):
            oc = oc | (self.yv[j] > 0)
        self.oc = oc

    def _one(self, lo, hi, oc):
        cons = self.IO(input_vars=self.xv, output_vars=self.yv, input_constraint=(self.xv >= lo) & (self.xv <= hi),
                       output_constraint=oc)
        r = self.solver.verify(constraints=cons)
        dom = 0
        try:
            dom = int(r.stats["bab"][0][2])
        except Exception:
            pass
        return r.status == "verified", dom, r.status

    def __call__(self, lo, hi, order=None):
        """input mode: the OR of the 9 specs in one call. relu / mixed modes crash on the 9-row OR spec in this
        alpha-beta-CROWN version, so they verify single specs 'margin_j > 0', for the top-2 j by the given order"""
        if self.mode == "input":
            return self._one(lo, hi, self.oc)
        doms = 0
        for j in (order[:2] if order is not None else range(2)):
            ok, dom, stt = self._one(lo, hi, self.yv[int(j)] > 0); doms += dom
            if ok:
                return True, doms, stt
        return False, doms, stt


def run(a):
    global SPLIT
    SPLIT = a.split
    tri = tuple(int(v) for v in a.tri.split(",")); stages = a.stages.split(",")
    B = Bounder(tri, a.alpha_iter); abc = ABC(tri, a.abc_timeout, a.abc_mode) if "abc" in stages else None
    th.set_default_dtype(th.float64)
    if a.resume:
        S = th.load(a.resume); LO, HI, SD = S["lo"], S["hi"], S["sd"]
    else:
        lo0, hi0 = net.root(tri); LO, HI = lo0[None].clone(), hi0[None].clone()
        lb, cu = B(LO, HI, "CROWN"); SD = splitdim(lb, cu, LO, HI)
    CH = net.radial_charge(LO, HI); QM = net.quad_mass(LO, HI)
    st = dict(evals=0, pruned=0, pr_crown=0, pr_alpha=0, pr_abc=0, alpha_calls=0, abc_calls=0, abc_dom=0, abc_status={})
    curve = []; t0 = time.process_time(); last = -1e9; it = 0
    res = []                                                       # reservoir of pruned boxes (soundness check)
    rng = np.random.default_rng(418)
    while len(LO) and time.process_time() - t0 < a.sec:
        k = min(a.K, len(LO)); idx = th.topk(QM if a.order == "quad" else CH, k).indices; keep = th.ones(len(LO), dtype=th.bool); keep[idx] = False
        blo, bhi, d = LO[idx], HI[idx], SD[idx]; LO, HI, CH, QM, SD = LO[keep], HI[keep], CH[keep], QM[keep], SD[keep]
        ar = th.arange(k); mid = (blo[ar, d] + bhi[ar, d]) / 2
        h1 = bhi.clone(); h1[ar, d] = mid; l2 = blo.clone(); l2[ar, d] = mid
        clo, chi = th.cat([blo, l2]), th.cat([h1, bhi]); n = len(clo); st["evals"] += n; it += 1
        lb, cu = B(clo, chi, "CROWN"); pr = (lb > 0).any(1); st["pr_crown"] += int(pr.sum())
        if "alpha" in stages and (~pr).any():
            s = th.nonzero(~pr)[:, 0]; lb2, cu2 = B(clo[s], chi[s], "alpha-CROWN"); st["alpha_calls"] += len(s)
            p2 = (lb2 > 0).any(1); st["pr_alpha"] += int(p2.sum()); pr[s] = p2
            better = lb2.max(1).values > lb[s].max(1).values; lb[s[better]] = lb2[better]; cu[s[better]] = cu2[better]
        if abc is not None and (~pr).any():
            for i in th.nonzero(~pr)[:, 0].tolist():
                ok, dom, stt = abc(clo[i], chi[i], th.argsort(lb[i], descending=True)); st["abc_calls"] += 1; st["abc_dom"] += dom
                st["abc_status"][stt] = st["abc_status"].get(stt, 0) + 1
                if ok:
                    pr[i] = True; st["pr_abc"] += 1
        st["pruned"] += int(pr.sum())
        if pr.any() and len(res) < 400:
            s = th.nonzero(pr)[:, 0]; s = s[th.as_tensor(rng.permutation(len(s))[:20])]; res.append((clo[s], chi[s]))
        sv = ~pr
        LO, HI = th.cat([LO, clo[sv]]), th.cat([HI, chi[sv]])
        CH = th.cat([CH, net.radial_charge(clo[sv], chi[sv])]); QM = th.cat([QM, net.quad_mass(clo[sv], chi[sv])])
        SD = th.cat([SD, splitdim(lb[sv], cu[sv], clo[sv], chi[sv])])
        el = time.process_time() - t0
        if el - last > a.log_sec or not len(LO) or el >= a.sec:
            last = el; row = dict(it=it, open=len(LO), cpu_s=el, radial=float(CH.sum()) + TAIL, quad=float(QM.sum()) + TAIL,
                                  **{k_: v for k_, v in st.items() if k_ != "abc_status"})
            curve.append(row)
            print(" ".join("%s %s" % (k_, ("%.3e" % v) if isinstance(v, float) else v) for k_, v in row.items()), flush=True)
    out = dict(tri=tri, stages=stages, K=a.K, alpha_iter=a.alpha_iter, abc_timeout=a.abc_timeout, abc_mode=a.abc_mode,
               resume=a.resume, order=a.order, split=a.split, tail=TAIL, final=curve[-1] if curve else None, abc_status=st["abc_status"], curve=curve)
    json.dump(out, open(a.out, "w"), indent=1)
    th.save(dict(lo=LO, hi=HI, sd=SD, ch=CH, qm=QM,
                 pruned_lo=th.cat([r[0] for r in res]) if res else None, pruned_hi=th.cat([r[1] for r in res]) if res else None),
            a.out.replace(".json", "_open.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tri"); ap.add_argument("out"); ap.add_argument("sec", type=float)
    ap.add_argument("--stages", default="crown,alpha"); ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--alpha-iter", type=int, default=20); ap.add_argument("--resume", default="")
    ap.add_argument("--abc-timeout", type=float, default=2.0); ap.add_argument("--abc-mode", default="mixed")
    ap.add_argument("--log-sec", type=float, default=60)
    ap.add_argument("--split", default="margin", help="margin (T414, pre-registered) or widest (deviation D3)")
    ap.add_argument("--order", default="radial", help="radial (pre-registered) or quad (deviation D1: near-exact mass)")
    run(ap.parse_args())
