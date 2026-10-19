"""T454 runner. usage: python run3.py t403 I  -> t403_<I>.json ; python run3.py target T0 -> target_<T0>.json (secondary)
Pilot (pre-registered): 1e7 draws, net on the top 5e5 by g23/||x||, 200 seeds (T436-size coverage)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dominating_is2")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
from run2 import dev3
from is2 import Net, find_points
import is3

PILOT = dict(n_draw=10_000_000, n_net=500_000, n_seed=200)


def main():
    mode = sys.argv[1]
    if mode == "target":
        from common import Wi, Wh, Wo
        net = Net(Wi, Wh, Wo); t0 = float(sys.argv[2]); P = find_points(net, 301, **PILOT)
        r = is3.estimate(net, P, t0, 401); out = os.path.join(HERE, f"target_{t0}.json")
    else:
        i = int(sys.argv[2]); pt = json.load(open(os.path.join(HERE, "t403_points.json")))[i]; net = dev3(pt["model"])
        P = find_points(net, 3000 + i, **PILOT); r = is3.estimate(net, P, pt["t0"], 4000 + 10 * i)
        r.update(model=pt["model"], index=i); out = os.path.join(HERE, f"t403_{i}.json")
    r.update(pilot_fpe=P["pilot_fpe"], descent_sec=P["descent_sec"], n_cone_seeds=P["n_cone_seeds"], best_r2=P["points"][0]["r2"])
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: r[k] for k in ("t0", "p", "se", "rel_se", "hits", "max_weight", "weight_cap", "C", "F", "n_points", "extended")}, default=float), flush=True)


if __name__ == "__main__":
    main()
