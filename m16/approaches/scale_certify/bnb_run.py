"""T426 step 2: event-guided branch and bound (T398b setting: relax2 depth 6, DELTA = t0/7.5) on one triple of one model,
to the deepest k reachable within a CPU cap. Survivors are kept in memory; each child is pruned as soon as it is made.
Soundness check (could fail): every MC hit of this triple must lie in a surviving cell at every k.
usage: python bnb_run.py TAG T0 p1,p2,p3 CAP_SEC [--kmax K] -> results/bnb_<TAG>_<tri>.json, data/t426/{surv,cert}_*.npz"""
import json
import os
import sys
import time

import numpy as np

import core

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "data", "t426")
MAXSURV = 60_000                                                   # memory guard (~2 GB process limit)


def main():
    tag, t0 = sys.argv[1], float(sys.argv[2]); tri = tuple(int(v) for v in sys.argv[3].split(",")); cap = float(sys.argv[4])
    core.set_model(tag); core.DELTA = t0 / 7.5; T = core.D
    kmax = int(sys.argv[sys.argv.index("--kmax") + 1]) if "--kmax" in sys.argv else T
    name = f"{tag}_{'-'.join(map(str, tri))}"
    import glob
    hx = [x for f in sorted(glob.glob(os.path.join(HERE, "results", f"mc_{tag}_{t0:g}_*.json"))) for x in json.load(open(f)).get("hit_x", [])
          if tuple(np.argsort(-np.array(x))[:3]) == tri]
    hx = np.array(hx) if hx else np.zeros((0, T))
    hs = core.run(hx) if len(hx) else None
    out = dict(tag=tag, H=core.H, T=T, t0=t0, delta=core.DELTA, depth=core.DEPTH, tri=tri, cap_sec=cap, mc_hits=len(hx), steps=[])
    cells = [(core.cone(tri), core.interior(tri), np.zeros((core.H, T)), b"")]
    t_start = time.time(); path = os.path.join(HERE, "results", f"bnb_{name}.json"); os.makedirs(DATA, exist_ok=True)
    for t in range(kmax):
        t1 = time.time(); keep = []; nch = 0; st = []; cst, cstart, cleaves = [], [], []; censored = False; npar = 0
        for cell in cells:
            for c in core.children(cell, t):
                nch += 1; lv = []
                sg = core.prune_stage(c[0], c[2], t + 1, tri, cert=lv); st.append(sg)
                if sg < 0:
                    keep.append(c)
                else:
                    cst.append(sg); cstart.append(len(cleaves)); cleaves.extend(core.gates_of(c[3], t + 1, f) for f in lv)
            npar += 1
            if time.time() - t_start > cap or len(keep) > MAXSURV:
                censored = True; break
        st = np.array(st); kept = {c[3] for c in keep}
        hk = core.prefix_keys(hs, t + 1) if hs is not None else []
        rec = dict(k=t + 1, parents_done=npar, parents=len(cells), children=nch, survivors=len(keep),
                   by_stage={g: int((st == g).sum()) for g in range(-1, 4)}, hits_lost=sum(h not in kept for h in hk) if not censored else None,
                   hit_prefixes=len(set(hk)), sec=round(time.time() - t1, 1), complete=not censored)
        out["steps"].append(rec); print(rec, flush=True)
        if cst:
            np.savez_compressed(os.path.join(DATA, f"cert_{name}_k{t + 1}.npz"), tri=np.array(tri), k=t + 1, stage=np.array(cst, np.int8),
                                start=np.array(cstart), leaves=np.array(cleaves, np.int8), delta=core.DELTA)
        if not censored:
            np.savez_compressed(os.path.join(DATA, f"surv_{name}_k{t + 1}.npz"), tri=np.array(tri), k=t + 1,
                                gates=np.array([core.gates_of(c[3], t + 1, {}) for c in keep], np.int8).reshape(-1, T, core.H), delta=core.DELTA)
        out["cpu_sec"] = round(time.time() - t_start, 1); json.dump(out, open(path, "w"), indent=1)
        cells = keep
        if censored or not keep:
            break


if __name__ == "__main__":
    main()
