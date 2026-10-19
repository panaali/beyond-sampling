"""T427 estimates (DEDUCTIVE): walkgen.Walker(H, T).est_with_flag on the pre-registered t grid of each T421 model.
usage: python run_est.py TAG [TAG ...] -> results/est_<TAG>.json"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import grid  # noqa: E402
import t421  # noqa: E402
import walkgen  # noqa: E402

if __name__ == "__main__":
    for tag in sys.argv[1:]:
        path = os.path.join(HERE, "results", f"est_{tag}.json")
        if os.path.exists(path):
            continue
        m, rec, h = t421.load(tag); H, T = rec["H"], rec["T"]; ts = grid.tgrid(T); t0 = time.time()
        W = walkgen.Walker(H, T)
        r, share, f, itmax = W.est_with_flag(m, ts, log=lambda s: print(tag, s, f"{time.time() - t0:.0f}s", flush=True))
        out = dict(tag=tag, H=H, T=T, sha256=h, acc=rec["acc"], C=f / 5948, itmax=itmax, sec=time.time() - t0,
                   pts={f"{t:.6f}": dict(t=t, est=r[t], P_cond=grid.p_gap(t, T), flag_det=share[t]) for t in ts})
        json.dump(out, open(path, "w"), indent=1)
        print(tag, "done", f"C {out['C']:.4g} itmax {itmax} {out['sec']:.0f}s",
              {f"{t:.2f}": f"{r[t] / grid.p_gap(t, T):.3g}" for t in ts}, flush=True)
