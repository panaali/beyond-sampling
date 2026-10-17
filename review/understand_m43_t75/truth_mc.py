"""T84 item 1: independent truth. Own forward pass (mine.fwd) and plain MC with 1e7 inputs per model vs exact.py,
on the clean zoo seeds 0-4 and fresh noisy copies (sigma 0.05 seed 30 copies 0-3; sigma 0.1 seed 40 and sigma 0.02
seed 50 copies 0-1). Also checks the own forward pass against the torch model.
usage: OMP_NUM_THREADS=1 python truth_mc.py  -> truth_mc.log / truth_mc.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import torch
import exact as EX
from mine import zoo_w, noisy_w, fwd, mc

N = 10_000_000


def main():
    import alg_zoo
    x = np.random.default_rng(5).standard_normal((100_000, 3))
    for s in range(5):
        tm = alg_zoo.zoo_2nd_argmax(4, 3, seed=s)
        with torch.no_grad():
            lt = tm(torch.tensor(x, dtype=torch.float32)).numpy()
        lm = fwd(zoo_w(s), x)
        print(f"zoo {s}: own fwd vs torch max |dlogit| {np.abs(lt - lm).max():.1e}, argmax agree {np.mean(lt.argmax(1) == lm.argmax(1)):.5f}")
    rows = []; t0 = time.time()
    for s in range(5):
        b = zoo_w(s)
        ms = [("clean", b)] + [(f"s0.05/30/{i}", m) for i, m in enumerate(noisy_w(b, 4, 0.05, 30))] \
            + [(f"s0.1/40/{i}", m) for i, m in enumerate(noisy_w(b, 2, 0.1, 40))] \
            + [(f"s0.02/50/{i}", m) for i, m in enumerate(noisy_w(b, 2, 0.02, 50))]
        for j, (name, m) in enumerate(ms):
            ex = EX.exact_acc(m); p = mc(m, N, 1000 + 100 * s + j)
            z = (p - ex) / np.sqrt(ex * (1 - ex) / N)
            rows.append(dict(zoo=s, model=name, exact=ex, mc=p, z=float(z)))
            print(f"zoo {s} {name:12s} exact {ex:.5f} mc {p:.5f} z {z:+.2f}", flush=True)
    z = np.array([r["z"] for r in rows])
    print(f"{len(z)} models: mean z {z.mean():+.2f}, rms z {np.sqrt(np.mean(z**2)):.2f}, max |z| {np.abs(z).max():.2f}; {time.time()-t0:.0f}s")
    json.dump(rows, open(os.path.join(HERE, "truth_mc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
