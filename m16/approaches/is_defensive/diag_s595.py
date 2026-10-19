"""POST-HOC diagnosis (not pre-registered, run after scoring): where is dev3_s595's E1(1.75) mass that v3 misses?
Defensive-only sampling (= gap-conditional MC, T403's truth sampler in law): hits by triple, vs the triples of the
mixture components used by run3 point #7, plus the min ||x||/g23 of hits per triple. usage: python diag_s595.py N"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dominating_is2")); sys.path.insert(0, HERE)
from run2 import dev3
from is2 import event, find_points, select
from is3 import Defensive
net = dev3("dev3_s595"); t0 = 1.75; d = Defensive(t0); rng = np.random.default_rng(777); N = int(float(sys.argv[1]))
hits = {}; best = {}
for s in range(0, N, 500_000):
    X = d.sample(rng, 500_000); h, top, _ = event(net, X, t0)
    for x, tr in zip(X[h], top[h]):
        k = str(tuple(int(i) for i in tr)); hits[k] = hits.get(k, 0) + 1
        top_, v = np.sort(x)[::-1][:3], None
        r = float(np.linalg.norm(x) / (np.sort(x)[-2] - np.sort(x)[-3])); best[k] = min(best.get(k, 1e9), r * r)
P = find_points(net, 3007, n_draw=10_000_000, n_net=500_000, n_seed=200); sel = select(P["points"], t0)
tot = sum(hits.values())
out = dict(N=N, p_cond=tot / N, p=tot / N * d.Z, hits=hits, min_r2_by_triple=best,
           mixture_triples=sorted({str(tuple(p["tri"])) for p in sel}), all_point_triples=sorted({str(tuple(p["tri"])) for p in P["points"]}))
json.dump(out, open(os.path.join(HERE, "diag_s595.json"), "w"), indent=1)
print("p ~ %.3e from %d hits" % (out["p"], tot))
for k, c in sorted(hits.items(), key=lambda kv: -kv[1]):
    print(k, c, "%.0f%%" % (100 * c / tot), "min r2 %.2f" % best[k], "IN MIXTURE" if k in out["mixture_triples"] else ("in point set, cut" if k in out["all_point_triples"] else "NOT FOUND"))
