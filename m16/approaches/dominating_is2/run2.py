"""T453 runner. usage:
  python run2.py check                 code check on dev3_s526 (EXCLUDED model; tiny n; not scored)
  python run2.py target 1.5            validation on M_16,10 -> target_1.5.json
  python run2.py t403 I                T403 point I (0-14, order of T403 results/score.json 'rows') -> t403_<I>.json
Weights: M_16,10 from m16/common.py; DEV3 from data/dev3/weights.npz (sha256 62c95203..., as T403 models.py)."""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np
from is2 import Net, find_points, estimate

DEV3 = os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")
POINTS = os.path.join(HERE, "t403_points.json")      # (model, t0) only: no truths


def dev3(name):
    assert hashlib.sha256(open(DEV3, "rb").read()).hexdigest().startswith("62c952037e8c9ec2")
    Z = np.load(DEV3); i = [str(s) for s in Z["names"]].index(name)
    return Net(*(Z[k][i].astype(np.float64) for k in ("Wi", "Wh", "Wo")))


def main():
    mode = sys.argv[1]
    if mode == "check":
        net = dev3("dev3_s526"); P = find_points(net, 1); r = estimate(net, P, 1.0, 11, n1=20_000, n2=20_000, extend=False)
        print({k: r[k] for k in ("p", "se", "hits", "n_points", "n_components", "C")}, P["points"][0]["r2"], P["points"][0]["tri"])
        return
    if mode == "target":
        from common import Wi, Wh, Wo
        net = Net(Wi, Wh, Wo); t0 = float(sys.argv[2]); P = find_points(net, 101)
        r = estimate(net, P, t0, 201); r["points_best"] = P["points"][:5]; out = os.path.join(HERE, f"target_{t0}.json")
    else:
        i = int(sys.argv[2]); pt = json.load(open(POINTS))[i]; net = dev3(pt["model"])
        P = find_points(net, 1000 + i); r = estimate(net, P, pt["t0"], 2000 + 10 * i)
        r.update(model=pt["model"], index=i); out = os.path.join(HERE, f"t403_{i}.json")
    r.update(pilot_fpe=P["pilot_fpe"], descent_sec=P["descent_sec"], n_cone_seeds=P["n_cone_seeds"])
    json.dump(r, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: r[k] for k in ("t0", "p", "se", "rel_se", "hits", "C", "F", "F_draws", "n_points", "n_components", "extended")}, default=float), flush=True)


if __name__ == "__main__":
    main()
