"""T417: cache real simplices (float vertex enclosures, triangulation, d, mlow, jp3, jp2) from the bench k=4 cells
and harvested 10-D event regions, as the seeds of the adversarial generators.
usage: python geom.py  -> data/geom.npz"""
import json
import os
import signal
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ECR = os.path.join(HERE, "..", "event_cone_rigour")
sys.path.insert(0, ECR)
import m5t_fast as F  # noqa: E402
import rigour as R  # noqa: E402


def main():
    out = []
    dat = np.load(os.path.join(ECR, "data/bench_182_k4.npz"), allow_pickle=True)
    tri = tuple(int(v) for v in dat["tri"]); k = int(dat["k"])
    cells = sorted({o["i"] for t0 in ("2.25", "2.5") for o in json.load(open(os.path.join(ECR, f"results_a_m5t_{t0}_2000.json")))})
    for i in cells:
        g = np.full((10, 16), -1); g[:k] = np.unpackbits(np.frombuffer(dat["hist"][i], np.uint8)).reshape(k, 16)
        gm = F.cell_geometry(tri, g, k)
        if gm is not None:
            out.append(("bench", i, tri, k, gm))
    reg = np.load(os.path.join(ECR, "data/regions_1-8-2.npz")); tri = tuple(int(v) for v in reg["tri"])
    rng = np.random.default_rng(417)

    def alarm(*a):
        raise TimeoutError
    signal.signal(signal.SIGALRM, alarm)
    for i in [3] + list(rng.choice(len(reg["gates"]), 12, replace=False)):
        signal.alarm(120)                               # skip regions whose exact geometry takes > 2 min
        try:
            gm = F.cell_geometry(tri, reg["gates"][i].astype(int), 10, event=True)
        except Exception as e:  # noqa: BLE001
            print("region", i, "failed", repr(e)); continue
        finally:
            signal.alarm(0)
        if gm is not None:
            out.append(("region", int(i), tri, 10, gm))
        print("region", i, None if gm is None else gm[5].shape, flush=True)
    res = {}
    for j, (kind, i, tri, k, gm) in enumerate(out):
        d, mlow, jp3, Mlo, Mhi, S, vlo, vhi, nV = gm
        I = sorted(set(range(k)) | set(tri))
        res[f"c{j}_Mlo"] = Mlo; res[f"c{j}_S"] = S
        res[f"c{j}_meta"] = np.array([d, mlow, jp3, I.index(tri[1]), k, i, kind == "region"])
    res["n"] = np.array(len(out))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    np.savez(os.path.join(HERE, "data/geom.npz"), **res)
    print("cells", len(out), "simplices", sum(len(o[4][5]) for o in out))


if __name__ == "__main__":
    main()
