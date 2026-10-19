"""T397: E33's tie-slicing recipe (ht_seeds/est.py est_faces + ht_seeds/core.py, UNCHANGED; hashes asserted), rule seed
700 (E33's first rule), equal faces per kind, scored on DEV3 (100 retrains, 1e8 truths) and on the M_16,10 leaderboard
models (orig, seeds 0-4, 24 noisy copies; leaderboard truths) with cost AS EXECUTED: that code recomputes the full
sequence (value + tangent) for every cell, so executed FPE = cells x (2 + 8H/(2H^2+2H)), the same per-cell overhead
convention as walk.py. The claimed cost (ht_seeds' own count) is reported alongside.
n per kind: 680 (16k) and 2720 (64k), fixed from DEV2 measurements (~23.5 executed FPE per face pair)."""
import hashlib, importlib.util, json, os, sys, time
import numpy as np
HT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ht_seeds")
H_CORE, H_EST = "a78a45c1fae015507fb7b6a9764064a9b2425f52ff599f378408640f0617170e", "b6154cad9a73eeb5b4468d50d08ab7cdae1da27d1d5698e759cd2c90d2646750"
assert hashlib.sha256(open(f"{HT}/core.py", "rb").read()).hexdigest() == H_CORE
assert hashlib.sha256(open(f"{HT}/est.py", "rb").read()).hexdigest() == H_EST
sys.path.insert(0, HT)
import core as Cht  # noqa: E402  (ht_seeds core: est.py imports it as "core")
import est as Eht   # noqa: E402
assert Cht.__file__.startswith(os.path.abspath(HT)) or os.path.dirname(os.path.abspath(Cht.__file__)) == os.path.abspath(HT)
NS = {16000: 680, 64000: 2720}
PER_CELL = lambda H: 2 + 8 * H / (2 * H * H + 2 * H)
_orig = Cht.walk_mass
_cells = []
def _wrap(*a, **k):
    m, c, cells = _orig(*a, **k); _cells.append(cells.sum()); return m, c, cells
Cht.walk_mass = _wrap
def run(model, b):
    _cells.clear(); t0 = time.process_time()
    acc, claimed, _ = Eht.est_faces(model, NS[b], NS[b], rule_seed=700)
    return acc, sum(_cells) * PER_CELL(model[1].shape[0]), claimed, time.process_time() - t0
panel = sys.argv[1]
out = []
if panel == "dev3":
    W = np.load("../../../data/dev3/weights.npz"); tr = json.load(open("../../../data/dev3/truths.json"))["truths"]
    mods = [(str(n), (W["Wi"][i], W["Wh"][i], W["Wo"][i]), tr[str(n)]["acc"], tr[str(n)]["se"]) for i, n in enumerate(W["names"])]
else:
    sys.path.insert(0, "../..")
    import leaderboard as LB
    tr = json.load(open("../../../results/leaderboard_truths.json"))
    mods = [(n, m, tr[n]["acc"], tr[n]["se"]) for n, m in LB.models(16, 10).items()]
for b in (16000, 64000):
    for name, m, p, se in mods:
        e, c, cl, cpu = run(tuple(np.asarray(w, float) for w in m), b)
        out.append(dict(panel=panel, model=name, budget=b, est=e, cost=c, claimed=cl, cpu=cpu, truth=p, se=se))
    json.dump(out, open(f"e33/scores_{panel}.json", "w"), indent=0)
    print(panel, b, "done", flush=True)
