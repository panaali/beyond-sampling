"""T424 estimates (DEDUCTIVE): gen.est_k3 and gen.est_graded at each model's two pre-registered thresholds.
usage: python est_run.py TAG [TAG ...] -> results/est/TAG.json (one file per model; skips models already done)"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gen  # noqa: E402
import models as M  # noqa: E402
import thresholds as TH  # noqa: E402

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results", "est"), exist_ok=True)
    for tag in sys.argv[1:]:
        fn = os.path.join(HERE, "results", "est", f"{tag}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
        row = M.by_tag(tag); m = M.load(row); t0s = TH.choose(row["T"], row["acc"]); F = gen.fpe(row["H"], row["T"])
        out.update(tag=tag, H=row["H"], T=row["T"], acc=row["acc"], t0s=t0s, fpe=F)
        for name, fun in (("k3", gen.est_k3), ("graded", gen.est_graded)):
            if name in out:
                continue
            t = time.time(); st = {}; r, f = fun(m, t0s, stats=st)
            out[name] = dict(est={str(k): float(v) for k, v in r.items()}, C_fpe=f / F, sec=time.time() - t, truncated_rows=st["truncated_rows"])
            json.dump(out, open(fn + ".tmp", "w"), indent=1); os.replace(fn + ".tmp", fn)
            print(tag, name, out[name], flush=True)
