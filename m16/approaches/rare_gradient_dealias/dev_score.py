"""T386 development table on dev3_s526 and the pre-registered freeze rule -> results/dev_s526.txt, results/frozen.json
(frozen.json is written only with --freeze, after the whole grid has run)."""
import glob
import json
import os
import sys

import numpy as np

import common

NM = "dev3_s526"; FINEST = ("mid20x16", "qmid3x8")


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main(freeze):
    T = common.R(f"truth_fd2_{NM}.json"); C = common.R(f"cmc_{NM}.json"); gc = np.array(C["grad"])
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(common.HERE, "results", f"cfg_{NM}_*.json"))]
    fin = [r for r in rows if (r["v2"], r["g12"]) == FINEST]; gf = np.array(fin[0]["grad"]) if fin else None
    lines = [f"{NM} E1({T['t0']}): truth P {T['P']:.4e}; truth FD/P " + " ".join(f"{f / T['P']:.2f}" for f in T["FD_sel"]),
             "v2        g12      nodes  est/P   pass  pred/P per dir (ratio)                                   cos_CMC cos_fin  C_walk/K3"]
    out = []
    for r in rows:
        M = int(r["v2"].split("x")[1]); ng = 3 * (int(r["g12"].split("x")[1]) if "x" in r["g12"] else 1)
        g = np.array(r["grad"]); js = [common.judge(p, f, s) for p, f, s in zip(r["pred"], T["FD_sel"], T["FD_sel_se"])]
        npass = sum(j[1] for j in js); cf = cos(g, gf) if gf is not None else float("nan")
        out.append(dict(v2=r["v2"], g12=r["g12"], nodes=40 * M * ng, npass=npass, cos_fin=cf, cos_cmc=cos(g, gc), est=r["est"],
                        C_walk=r["C_walk"], C_grad=r["C_grad"]))
        lines.append(f"{r['v2']:9} {r['g12']:8} {40 * M * ng:6d} {r['est'] / T['P']:5.2f}  {npass}/6   "
                     + " ".join(f"{p / T['P']:6.1f}({j[2]:5.2f}{'Y' if j[1] else 'n'})" for p, j in zip(r["pred"], js))
                     + f"  {cos(g, gc):.3f}  {cf:.3f}  {r['C_walk'] / 325201:.1f}")
    out.sort(key=lambda o: (o["nodes"], -o["cos_fin"])); order = {(o["v2"], o["g12"]): i for i, o in enumerate(out)}
    lines = lines[:2] + sorted(lines[2:], key=lambda s: order[tuple(s.split()[:2])])
    ok = [o for o in out if o["npass"] >= 5 and o["cos_fin"] >= 0.95]
    pick = ok[0] if ok else [o for o in out if (o["v2"], o["g12"]) == FINEST][0]
    lines.append(f"freeze rule -> {pick['v2']} / {pick['g12']} ({'meets (a)+(b)' if ok else 'none meets (a)+(b): finest'})")
    txt = "\n".join(lines); print(txt); open(os.path.join(common.HERE, "results", "dev_s526.txt"), "w").write(txt + "\n")
    if freeze:
        json.dump(dict(pick, rule_met=bool(ok), n_configs=len(out)), open(os.path.join(common.HERE, "results", "frozen.json"), "w"), indent=1)


if __name__ == "__main__":
    main("--freeze" in sys.argv)
