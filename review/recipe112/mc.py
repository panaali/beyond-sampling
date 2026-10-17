"""T113 item 1: independent truth check. Own forward pass vs the torch zoo model (argmax agreement), then 1e7-sample
MC (own forward, own rng) against arc_m43_repro/exact.py and against own exact_mine, on the clean zoo models of
(4,3), (5,3), (6,3), (8,3), (16,3) and one fresh noisy copy (seed 80, copy 0) of each of rungs 4-8.
usage: OMP_NUM_THREADS=1 python mc.py  -> out/mc.log, out/mc.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import torch
import exact as EX
import mine as M

N = 10_000_000


def main():
    import alg_zoo
    x = np.random.default_rng(5).standard_normal((100_000, 3))
    for H in (4, 8, 16):
        for s in range(5):
            tm = alg_zoo.zoo_2nd_argmax(H, 3, seed=s)
            with torch.no_grad():
                lt = tm(torch.tensor(x, dtype=torch.float32)).numpy()
            lm = M.fwd(M.zoo_w(H, s), x)
            print(f"H={H} zoo {s}: own fwd vs torch argmax agree {np.mean(lt.argmax(1) == lm.argmax(1)):.5f}", flush=True)
    rows = []; t0 = time.time()
    for H in (4, 5, 6, 8, 16):
        for s in range(5):
            b = M.zoo_w(H, s)
            ms = [("clean", b)] + ([("abs0.05/80/0", M.noisy_abs(b, 1, 0.05, 80)[0])] if H != 16 else [])
            for name, m in ms:
                ex = EX.exact_acc(m); em = M.exact_mine(m); p = M.mc(m, N, 4000 + 100 * H + 10 * s + len(rows))
                z = (p - ex) / np.sqrt(max(ex * (1 - ex), 1e-12) / N)
                rows.append(dict(H=H, zoo=s, model=name, exact=ex, mine=em, mc=p, z=float(z)))
                print(f"H={H} zoo {s} {name:13s} exact.py {ex:.6f} own exact {em:.6f} (diff {em-ex:+.1e}) MC {p:.6f} z {z:+.2f}", flush=True)
    z = np.array([r["z"] for r in rows])
    print(f"{len(z)} models: mean z {z.mean():+.2f}, rms z {np.sqrt(np.mean(z**2)):.2f}, max |z| {np.abs(z).max():.2f}; "
          f"max |own - exact.py| {max(abs(r['mine']-r['exact']) for r in rows):.1e}; {time.time()-t0:.0f}s")
    json.dump(rows, open(os.path.join(HERE, "out", "mc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
