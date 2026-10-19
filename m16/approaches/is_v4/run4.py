"""T456 runner. usage: python run4.py est I -> est_<I>.json ; python run4.py truth I -> truth_<I>.json (ONLY after all
estimates are pushed). Panel = panel.json (pre-registered)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "dominating_is2"), os.path.join(HERE, "..", "is_defensive"), HERE]
import numpy as np
from run2 import dev3
from is2 import find_points, event
import is4
from is3 import Defensive

PILOT = dict(n_draw=10_000_000, n_net=500_000, n_seed=200)
PANEL = json.load(open(os.path.join(HERE, "panel.json")))


def main():
    mode, i = sys.argv[1], int(sys.argv[2]); pt = PANEL[i]; net = dev3(pt["model"])
    if mode == "est":
        P = find_points(net, 5000 + i, **PILOT); r = is4.estimate(net, P, pt["t0"], 6000 + 10 * i)
        r.update(pt, index=i, pilot_fpe=P["pilot_fpe"], descent_sec=P["descent_sec"]); out = f"est_{i}.json"
        keys = ("p", "se", "rel_se", "hits", "flag", "gate_maxw", "gate_capshare", "n_capped", "C", "F", "n_new_descents", "extended")
    else:
        assert all(os.path.exists(os.path.join(HERE, f"est_{j}.json")) for j in range(len(PANEL))), "estimates first"
        d = Defensive(pt["t0"]); rng = np.random.default_rng(7000 + i); n = h = 0; t = time.time()
        while h < 1000 and n < 2_000_000_000:
            X = d.sample(rng, 1_000_000); hit, _, _ = event(net, X, pt["t0"]); h += int(hit.sum()); n += len(X)
        p = d.Z * h / n; r = dict(pt, index=i, truth=p, se=d.Z * np.sqrt(h) / n, hits=h, n=n, Z=d.Z, sec=time.time() - t)
        out = f"truth_{i}.json"; keys = ("truth", "se", "hits", "n")
    json.dump(r, open(os.path.join(HERE, out), "w"), indent=1, default=float)
    print(i, json.dumps({k: r[k] for k in keys}, default=float), flush=True)


if __name__ == "__main__":
    main()
