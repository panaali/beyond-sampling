"""Summarise results/meanfield_oracle.json per group: surrogate - network (pp, mean / RMS over models), split into
blind (K<=1) and bulk classes, per-class RMS, and the between-model correlation of each surrogate's accuracy (and of
its blind-region accuracy contribution) with the network's. -> results/meanfield_summary.txt"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402

r = json.load(open(os.path.join(cf.HERE, "results", "meanfield_oracle.json")))
L = []
for g in cf.GROUPS + cf.DEV2_GROUPS + ("DEV2 all",):
    ms = [m for m in r if (cf.group_of(m) in cf.DEV2_GROUPS if g == "DEV2 all" else cf.group_of(m) == g)]
    if not ms:
        continue
    net = np.array([r[m]["net"] for m in ms]); line = f"{g:9s} n={len(ms):2d} net {net.mean():.4f}"
    for s in ("frozen", "gauss"):
        e = 100 * (np.array([r[m][s] for m in ms]) - net)
        bl = 100 * np.array([r[m][f"blind_bias_{s}"] for m in ms]); bu = 100 * np.array([r[m][f"bulk_bias_{s}"] for m in ms])
        cr = np.corrcoef(net, [r[m][s] for m in ms])[0, 1] if len(ms) > 2 else np.nan
        line += (f" | {s}: {e.mean():+.2f}/{np.sqrt((e ** 2).mean()):.2f} (blind {bl.mean():+.2f}, bulk {bu.mean():+.2f})"
                 f" cls_rms {np.mean([r[m]['cls_rms_' + s] for m in ms]):.3f} corr {cr:+.2f}")
    line += f" | noise cls_rms {np.mean([r[m]['cls_rms_noise'] for m in ms]):.3f}"
    L.append(line)
open(os.path.join(cf.HERE, "results", "meanfield_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
