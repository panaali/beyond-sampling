"""T386 development table on dev3_s526 and the pre-registered freeze rule -> results/dev_s526.txt, results/frozen.json
(frozen.json is written only with --freeze, after the whole grid has run)."""
import glob
import json
import os
import sys

import numpy as np

import common

NM = "dev3_s526"; FINEST = {"A": ("mid20x16", "qmid3x8"), "T": ("T8", "mid20x4")}      # amendment A1
fam = lambda r: "T" if r["v2"].startswith("T") else "A"


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main(freeze):
    T = common.R(f"truth_fd2_{NM}.json"); C = common.R(f"cmc_{NM}.json"); gc = np.array(C["grad"])
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(common.HERE, "results", f"cfg_{NM}_*.json"))]
    gf = {f: np.array(r["grad"]) for r in rows for f, k in FINEST.items() if (r["v2"], r["g12"]) == k}
    lines = [f"{NM} E1({T['t0']}): truth P {T['P']:.4e}; truth FD/P " + " ".join(f"{f / T['P']:.2f}" for f in T["FD_sel"]),
             "v2        g12      lines/720  est/P   pass  pred/P per dir (ratio)                                   cos_CMC cos_fin  C_walk/K3"]
    out = []
    for r in rows:
        if fam(r) == "T":
            M = int(r["g12"].split("x")[1]) if "x" in r["g12"] else 1; ng = int(r["v2"][1:])
        else:
            M = int(r["v2"].split("x")[1]); ng = 3 * (int(r["g12"].split("x")[1]) if "x" in r["g12"] else 1)
        g = np.array(r["grad"]); js = [common.judge(p, f, s) for p, f, s in zip(r["pred"], T["FD_sel"], T["FD_sel_se"])]
        npass = sum(j[1] for j in js); cf = cos(g, gf[fam(r)]) if fam(r) in gf else float("nan")
        out.append(dict(family=fam(r), v2=r["v2"], g12=r["g12"], nodes=40 * M * ng, npass=npass, cos_fin=cf, cos_cmc=cos(g, gc), est=r["est"],
                        C_walk=r["C_walk"], C_grad=r["C_grad"]))
        lines.append(f"{r['v2']:9} {r['g12']:8} {40 * M * ng:6d} {r['est'] / T['P']:5.2f}  {npass}/6   "
                     + " ".join(f"{p / T['P']:6.1f}({j[2]:5.2f}{'Y' if j[1] else 'n'})" for p, j in zip(r["pred"], js))
                     + f"  {cos(g, gc):.3f}  {cf:.3f}  {r['C_walk'] / 325201:.1f}")
    out.sort(key=lambda o: (o["C_walk"], -o["cos_cmc"])); order = {(o["v2"], o["g12"]): i for i, o in enumerate(out)}
    lines = lines[:2] + sorted(lines[2:], key=lambda s: order[tuple(s.split()[:2])])
    assert not freeze or all(k in gf for k in FINEST), "freeze needs both finest configs"
    ok = [o for o in out if o["npass"] >= 5 and o["cos_fin"] >= 0.95]
    conv = [o for o in out if o["cos_fin"] >= 0.95]
    if ok:
        pick, why = ok[0], "cheapest meeting (a)+(b)"
    elif conv:
        pick, why = sorted(conv, key=lambda o: (-o["npass"], -o["cos_cmc"]))[0], "none meets (a)+(b): most passes among (b)"
    else:
        fT = [o for o in out if (o["v2"], o["g12"]) == FINEST["T"]]
        pick, why = (fT[0] if fT else dict(v2="?", g12="?")), "none meets (b): finest T"
    lines.append(f"freeze rule (A1) -> {pick['v2']} / {pick['g12']} ({why})")
    txt = "\n".join(lines); print(txt); open(os.path.join(common.HERE, "results", "dev_s526.txt"), "w").write(txt + "\n")
    if freeze:
        json.dump(dict(pick, why=why, n_configs=len(out)), open(os.path.join(common.HERE, "results", "frozen.json"), "w"), indent=1)


if __name__ == "__main__":
    main("--freeze" in sys.argv)
