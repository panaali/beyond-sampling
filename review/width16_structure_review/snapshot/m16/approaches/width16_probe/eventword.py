"""T217 (memo D1): is the prefix region growth mass-irrelevant given the mechanism's event word?
Exact prefix leaves (prefix.run: cone in R^k + full gate history) are split by every order hyperplane x_i = x_j
(estT.clip), so each piece has one ordering of x_0..x_{k-1} and hence one event word: for t = 1..k-1, x_t is a new
record R (above all earlier inputs), a new 2nd S (below the running max, above all other earlier inputs) or low L
(x_0 is always the first record and is left out). Exact Gaussian masses of the pieces (estT.mass, d = k; diagnostic,
not an estimator cost). With p(e, g) the mass of event word e and gate word g:
  branching B_k = exp(H(G | E)) (mass-weighted, nats); per-step factor B_k / B_{k-1};
  modal share M_k = sum_e max_g p(e, g) (mass of the most likely gate word within each event word).
Variant *_sign0: the event word extended by sign(x_0). Also exp(H(G)) unconditional, the number of gate words carrying 99% of each event word's mass, and sum p - 1.
usage: python eventword.py model k   model = target | zoo16_4 | zoo16_5 (zoo_2nd_argmax(16, T, 0), first k steps)
-> eventword/<model>_k{k}.json + .npz (per piece: event word, gate-word id, mass)"""
import json, os, sys, time
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops
from prefix import target, run
from zoo44 import zoo

name, k = sys.argv[1], int(sys.argv[2])
model = target() if name == "target" else zoo(16, int(name[-1]), 0)
fn = os.path.join(HERE, "eventword", f"{name}_k{k}")
t0 = time.process_time(); r, leaves = run(model, k, save=True); E = np.eye(k)
PAIRS = [(i, j) for i in range(k) for j in range(i + 1, k)]


def event(x):
    w = []
    for t in range(1, k):
        prev = np.sort(x[:t])[::-1]
        w.append("R" if x[t] > prev[0] else ("S" if t == 1 or x[t] > prev[1] else "L"))
    return "".join(w)


P = defaultdict(float); gid = {}; ev, gg, mm, SX = [], [], [], []
for c, hist in leaves:
    pieces = [c]
    for i, j in PAIRS:
        nxt = []
        for q in pieces:
            for part in estT.clip(q, E[i] - E[j], Ops()):
                if part is not None:
                    nxt.append(part)
        pieces = nxt
    g = gid.setdefault(hist, len(gid))
    for q in pieces:
        x = q.R.mean(0); w = estT.mass(q, Ops()); e = event(x); P[(e, g)] += w; ev.append(e); gg.append(g); mm.append(w)
        SX.append('+' if x[0] > 0 else '-')
pe = defaultdict(float); pg = defaultdict(float); best = defaultdict(float); per_e = defaultdict(list)
for (e, g), w in P.items():
    pe[e] += w; pg[g] += w; best[e] = max(best[e], w); per_e[e].append(w)
tot = sum(pe.values())
Ps = defaultdict(float); pes = defaultdict(float); bests = defaultdict(float)     # variant: event word + sign(x_0)
for e, g, w, sx in zip(ev, gg, mm, SX):
    Ps[(sx + e, g)] += w
for (e, g), w in Ps.items():
    pes[e] += w; bests[e] = max(bests[e], w)
HGEs = -sum(w * np.log(w / pes[e]) for (e, g), w in Ps.items() if w > 0) / tot
HGE = -sum(w * np.log(w / pe[e]) for (e, g), w in P.items() if w > 0) / tot
HG = -sum(w * np.log(w / tot) for w in pg.values() if w > 0) / tot
n99 = {}
for e, ws in per_e.items():
    ws = np.sort(ws)[::-1]; n99[e] = int(np.searchsorted(np.cumsum(ws), 0.99 * pe[e]) + 1)
out = dict(r, model=name, pieces=len(mm), mass_sum_minus_1=tot - 1, event_words=len(pe), eg_words=len(P),
           branching_given_event=float(np.exp(HGE)), branching_unconditional=float(np.exp(HG)),
           modal_share=sum(best.values()) / tot,
           branching_given_event_sign0=float(np.exp(HGEs)), modal_share_sign0=sum(bests.values()) / tot, n99_mass_weighted=sum(pe[e] * n99[e] for e in pe) / tot,
           per_event={e: dict(mass=pe[e], modal=best[e] / pe[e], gate_words=len(per_e[e]), n99=n99[e]) for e in sorted(pe)},
           cpu_total=time.process_time() - t0)
os.makedirs(os.path.dirname(fn), exist_ok=True); json.dump(out, open(fn + ".json", "w"), indent=1)
np.savez_compressed(fn + ".npz", event=np.array(ev), gate=np.array(gg), mass=np.array(mm))
print(name, k, {q: out[q] for q in ("regions", "pieces", "mass_sum_minus_1", "event_words", "branching_given_event",
                                     "branching_unconditional", "modal_share", "branching_given_event_sign0", "modal_share_sign0", "n99_mass_weighted", "cpu_total")}, flush=True)
