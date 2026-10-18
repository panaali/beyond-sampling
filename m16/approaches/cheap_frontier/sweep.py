"""Run the pre-registered cheap-settings grid on the LB3 panel. Resumable; one row per (family, cfg, model).
usage: python sweep.py FAMILY [--models dev2]   -> results/sweep_raw.jsonl"""
import itertools
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402


def grid(fam):
    if fam == "rx":
        return [dict(Q=Q, stride=s, kgroups=k) for Q in (4, 8, 16, 64) for s in (1, 2, 4, 8, 16) for k in (11, 3, 1)]
    if fam == "obs":
        return [dict(q=q, nq=nq, rbins=rb, stride=s) for q in (1, 2) for nq in (6, 8, 12) for rb in (6, 3)
                for s in (1, 3, 8, 24)]
    if fam == "probe":
        return ([dict(version=v, iters=i, stride=s) for v in (1, 2) for i in (4, 8, 12) for s in (1, 3, 8)]
                + [dict(version=3, iters=i, stride=s) for i in (8, 12) for s in (3, 8)])
    if fam == "rel":
        import rel
        return [dict(variant=v) for v in rel.VARIANTS]
    if fam == "lb":
        return [dict(name=n, budget=b) for n in ("cone_enum", "branch_bound", "propagation") for b in (1000,)]
    if fam == "v5":
        return [dict(which=w, budget=b) for w in ("v5a", "v5b", "v5q") for b in (1000, 4000)]
    if fam == "tcone":
        return [dict(m=m) for m in (0, 1, 2)]
    raise ValueError(fam)


def fn(fam):
    if fam == "rx":
        import rx; return rx.est_rx
    if fam == "obs":
        import obs; return obs.est_obs
    if fam == "probe":
        import probe; return probe.est_probe
    if fam == "rel":
        import rel; return rel.est_rel
    if fam in ("lb", "v5", "tcone"):
        import ext_fams
        return {"lb": ext_fams.lb_est, "v5": ext_fams.v5_est, "tcone": ext_fams.tcone_bias}[fam]


def main():
    fam = sys.argv[1]
    panel = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--models" else "lb3"
    if panel == "lb3":
        ms = cf.lb3_models(); out = os.path.join(cf.HERE, "results", "sweep_raw.jsonl")
    else:
        ms = cf.dev2_models(); out = os.path.join(cf.HERE, "results", f"sweep_raw_{panel}.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            d = json.loads(line); done.add((d["fam"], json.dumps(d["cfg"], sort_keys=True), d["model"]))
    f = fn(fam)
    truth = (cf.lb3_truths() if panel == "lb3" else cf.dev2_truths()) if fam == "tcone" else None
    with open(out, "a") as fo:
        for cfg in grid(fam):
            ck = json.dumps(cfg, sort_keys=True); t0 = time.time()
            for name, m in ms.items():
                if (fam, ck, name) in done:
                    continue
                t = time.process_time()
                kw = dict(name=name, truth=truth[name]["acc"]) if fam == "tcone" else {}
                e, c, parts = f(m, return_parts=True, **cfg, **kw)
                fo.write(json.dumps(dict(fam=fam, cfg=cfg, model=name, est=float(e), cost_fpe=float(c),
                                         cpu_s=time.process_time() - t,
                                         parts={k: float(v) for k, v in parts.items()})) + "\n")
                fo.flush()
            print(fam, ck, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
