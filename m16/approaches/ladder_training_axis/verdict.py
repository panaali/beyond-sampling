"""T101: pass/fail of a rung (ladder protocol): a recipe passes iff on EVERY zoo seed its panel MSE ratio < 1 and its
clean-model ratio < 1. usage: python verdict.py k  -> prints and exits 0 if both recipes pass, 1 if any fails."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def verdict(k):
    r = json.load(open(os.path.join(HERE, f"rung_k{k}.json")))
    out = {}
    for nm in ("t75", "f2l3"):
        fails = [s for s, d in r["seeds"].items() if d["panel"][nm]["ratio"] >= 1 or d["clean"][nm + "_ratio"] >= 1]
        out[nm] = (len(r["seeds"]) == 5 and not fails, fails)
    return out


if __name__ == "__main__":
    v = verdict(int(sys.argv[1])); print(sys.argv[1], v)
    sys.exit(0 if all(p for p, _ in v.values()) else 1)
