"""T414 diagnostics on a run's saved boxes: (b) soundness - uniform points in the reservoir of pruned boxes must contain
no event; pre-registered alternatives - for the top-charge open boxes, CROWN best lower bound vs the sampled minimum of
that spec (looseness), and whether a sampled point in the box is an event (true event mass); open-box widths.
usage: python diagnose.py p1,p2,p3 RUN_boxes.pt [n_open]"""
import sys

import numpy as np
import torch as th

import ibab

th.set_num_threads(1)
tri = tuple(int(v) for v in sys.argv[1].split(",")); p3 = tri[2]; P, q, _, _ = ibab.coords(tri)
B = th.load(sys.argv[2]); rng = np.random.default_rng(4141); nop = int(sys.argv[3]) if len(sys.argv) > 3 else 200


def sample(lo, hi, n):
    return lo + (hi - lo) * th.as_tensor(rng.uniform(0, 1, (n, 9)))


def margins_at(u):
    m = ibab.forward_margin(u @ P.T + q, p3); return th.cat([m[:, :p3], m[:, p3 + 1:]], 1)


rl, rh = B["pruned_lo"], B["pruned_hi"]; ev = 0; npts = 0
if rl is not None:
    per = max(1, 1000000 // len(rl))
    for i in range(len(rl)):
        m = margins_at(sample(rl[i], rh[i], per)); ev += int((m <= 0).all(1).sum()); npts += per
print("(b) soundness: %d points in %d pruned boxes, events found: %d -> %s" % (npts, 0 if rl is None else len(rl), ev, "PASS" if ev == 0 else "FAIL"))
lo, hi, ch = B["open_lo"], B["open_hi"], B["open_ch"]; top = th.topk(ch, min(nop, len(ch))).indices
lb, _ = ibab.margins(lo[top], hi[top], P, q, p3); gaps = []; evbox = 0; evch = 0.0
for i, b in enumerate(top):
    m = margins_at(sample(lo[b], hi[b], 2000)); j = int(lb[i].argmax())
    gaps.append(float(m[:, j].min() - lb[i, j])); e = bool((m <= 0).all(1).any()); evbox += e; evch += float(ch[b]) * e
w = (hi - lo)[top]
print("open boxes %d, total charge %.3e; top %d by charge hold %.3e" % (len(ch), float(ch.sum()), len(top), float(ch[top].sum())))
print("  with a sampled event point: %d / %d (their charge %.3e)" % (evbox, len(top), evch))
print("  CROWN looseness (sampled min - lb, best spec): median %.2f p10 %.2f p90 %.2f; best lb median %.2f" % (
    np.median(gaps), np.quantile(gaps, .1), np.quantile(gaps, .9), float(lb.max(1).values.median())))
print("  sampled best-spec min > 0 (prunable in principle): %d / %d" % (sum(g + float(lb[i].max()) > 0 for i, g in enumerate(gaps)), len(gaps)))
print("  open-box widths (top): median per dim", np.round(w.median(0).values.numpy(), 3).tolist())
