"""T385 scored estimator run (sampling-class): rank-r coordinate-line conditional MC (lines.py), N lines, seed
SEED[model] + 10 * r. Stores the per-hit-line values (Z/A p_gap and the 432-gradient) so every projection and
its s.e. (per-line, delta method for ratios) can be recomputed. usage: python est.py model rank N
-> data/t385/est_<model>_r<rank>.npz and results/est_<model>_r<rank>.json (summary; checkpointed)"""
import json, os, sys, time
import numpy as np
import core, lines

SEED = {"orig": 385200, "dev3_s528": 385201}
FPE = 5948.0          # FLOPs per forward pass of M_16,10 (repo convention)


def main(nm, r, N, chunk=200_000):
    model = core.load(nm); law = core.CondLaw(core.CASES[nm]); pr = np.zeros(10); pr[r - 1] = 1
    rng = np.random.default_rng(SEED[nm] + 10 * r); zs, gs = [], []; n = 0; ends = 0; cells = 0; t = time.time()
    while n < N:
        o = lines.run_chunk(model, law, rng, min(chunk, N - n), pr)
        zs.append(o["zval"]); gs.append(o["grad"]); n += o["n"]; ends += o["n_end"]; cells += o["cells"]
        if n % 2_000_000 == 0 or n >= N:
            z = np.concatenate(zs); G = np.concatenate(gs)
            np.savez_compressed(os.path.join(core.ROOT, "data", "t385", f"est_{nm}_r{r}.npz"), z=z, G=G, n=n)
            P = z.sum() / n; se = np.sqrt(max((z ** 2).sum() / n - P ** 2, 0) / n)
            # cost: each cell = one forward + one tangent pass (2 FPE) + root search (~.3 FPE); endpoint = fwd + bwd (3 FPE)
            js = dict(model=nm, t0=law.t0, rank=r, n_lines=n, n_hit=len(z), n_end=ends, cells=cells, P=P, P_se=se,
                      p_gap=law.p_gap, cost_fpe=2.3 * cells + 3 * ends, sec=time.time() - t)
            json.dump(js, open(f"results/est_{nm}_r{r}.json", "w"), indent=1)
            print(nm, r, n, "P %.4e +- %.1e" % (P, se), "%.0fs" % js["sec"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])))
