"""Implementation check (not scored): on fixed lines (same seed => same x, ranks), the analytic boundary gradient of
the estimate sum_i Z_i/A_i along a random direction u vs a central finite difference of the same estimate at W +- d u.
usage: python check_fd.py model n -> results/check_fd_<model>.json"""
import json, os, sys
import numpy as np
import core, lines
nm = sys.argv[1]; n = int(float(sys.argv[2])); model = core.load(nm); law = core.CondLaw(core.CASES[nm]); w0 = core.flat(model)
pr = np.full(4, .25).tolist() + [0] * 6; pr = np.array(pr)
est = lambda m: lines.run_chunk(m, law, np.random.default_rng(385800), n, pr)
o = est(model); out = dict(model=nm, n=n, n_hits=len(o["zval"]), n_end=o["n_end"], rows=[])
for d in range(4):
    u = np.random.default_rng(385810 + d).standard_normal(432); u /= np.linalg.norm(u)
    an = float((o["grad"] @ u).sum() / n)
    for dl in (1e-5, 1e-6, 1e-7):
        a = est(core.unflat(w0 + dl * u)); b = est(core.unflat(w0 - dl * u))
        fd = (a["zval"].sum() - b["zval"].sum()) / n / (2 * dl)
        out["rows"].append(dict(dir=d, delta=dl, analytic=an, fd=fd, rel=abs(fd - an) / abs(an)))
        print(nm, d, dl, "analytic %.6e fd %.6e rel %.2e" % (an, fd, abs(fd - an) / abs(an)), flush=True)
os.makedirs("results", exist_ok=True); json.dump(out, open(f"results/check_fd_{nm}.json", "w"), indent=1)
