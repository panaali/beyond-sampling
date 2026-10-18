"""Score results/oracle.jsonl: per closure x verdict, closure error per model = (verdict mean - network mean, paired)
+ (network sample mean - official T284 truth); RMS / mean over models, blind (v2 <= 0) and non-blind split, and the
added cost per the pre-registered model. -> results/score.json, printed table"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cheap_frontier"))
import cf  # noqa: E402

TR = cf.dev2_truths()
VERD = ("hard", "smooth", "smooth_prod", "clip", "clip_prod", "walk1d")
FWD1 = 1.0        # FPE of one hard verdict (one forward pass)
# extra FPE per closure point beyond the hard verdict (pre-registered cost model)
EXTRA = {"hard": lambda c: 0.0, "smooth": lambda c: 1.5, "smooth_prod": lambda c: 9 * 1.5,
         "clip": lambda c: 8.5, "clip_prod": lambda c: 8.5 + 9 * 1.5, "walk1d": lambda c: 1.5 + 2.0 * c}


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "oracle.jsonl"))]
    out = {}
    print(f"n models {len(rows)}; network sample - truth RMS "
          f"{100 * np.sqrt(np.mean([(r['mean']['net'] - TR[r['model']]['acc']) ** 2 for r in rows])):.3f} pp")
    print(f"{'closure:verdict':18s} {'RMS pp':>7s} {'mean':>7s} {'blind mean':>10s} {'nonbl mean':>10s} {'nonbl RMS':>9s} "
          f"{'cells':>6s} {'+FPE/triple':>11s}")
    for c in ("R0", "top4", "R1W"):
        cells = np.mean([r["mean"][f"{c}:_cells"] for r in rows])
        for v in VERD:
            k = f"{c}:{v}"
            e = np.array([r["mean"][k] - TR[r["model"]]["acc"] for r in rows])
            eb = np.array([r["blind_sum"][k] - r["blind_sum"]["net"] for r in rows])
            en = np.array([(r["mean"][k] - r["blind_sum"][k]) - (r["mean"]["net"] - r["blind_sum"]["net"]) for r in rows])
            add = 720 * EXTRA[v](cells)
            out[k] = dict(rms_pp=100 * np.sqrt((e ** 2).mean()), mean_pp=100 * e.mean(), blind_mean_pp=100 * eb.mean(),
                          nonblind_mean_pp=100 * en.mean(), nonblind_rms_pp=100 * np.sqrt((en ** 2).mean()),
                          cells=cells, added_fpe_per_triple=add)
            o = out[k]
            print(f"{k:18s} {o['rms_pp']:7.3f} {o['mean_pp']:+7.3f} {o['blind_mean_pp']:+10.3f} {o['nonblind_mean_pp']:+10.3f} "
                  f"{o['nonblind_rms_pp']:9.3f} {cells if v == 'walk1d' else 0:6.1f} {add:11.0f}")
    json.dump(out, open(os.path.join(HERE, "results", "score.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
