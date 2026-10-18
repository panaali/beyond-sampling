"""T367 truth-blind selection (pre-registered a9a57359): candidates from the K3 grid (results/t367_k3_grid.json), the frozen
fix (@5b4a1250 estimator_t366, hashes checked in T366) run on them, disagreement d = |log(fix/K3)|, top 4 models x 2 pairs.
-> results/t367_fix.json, results/t367_selection.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = "/tmp/claude-0/-home-user-clean-neurons/18e70fa5-7ff5-5fac-af53-4fea2351128f/scratchpad"
os.environ["T340_SRC"] = os.path.join(S, "src620")
sys.path.insert(0, os.path.join(S, "r1c366", "m16", "approaches", "r1_cont"))
import estimator_t366 as E  # noqa: E402

k3 = json.load(open(os.path.join(HERE, "results", "t367_k3_grid.json")))
z = np.load(os.path.join(HERE, "ext", "dev3", "weights.npz")); names = list(map(str, z["names"]))
cand = {}
for m, d in k3.items():
    est = {float(t): v for t, v in d["est"].items()}
    c = [t for t, v in est.items() if 1e-9 <= v <= 1e-8]
    if len(c) < 2:
        c = sorted(est, key=lambda t: abs(np.log10(max(est[t], 1e-300)) - np.log10(3e-9)))[:2]
    cand[m] = sorted(c)
fix = {}
for m, ts in cand.items():
    i = names.index(m); mod = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
    t = time.time(); r, f = E.est(mod, tuple(ts))
    fix[m] = dict(est={str(k): float(v) for k, v in r.items()}, C=f / 5948.0, sec=time.time() - t)
    print(m, ts, fix[m], flush=True)
    json.dump(fix, open(os.path.join(HERE, "results", "t367_fix.json"), "w"), indent=1)
score = {}
for m, ts in cand.items():
    d = sorted(((abs(np.log(fix[m]["est"][str(t)] / k3[m]["est"][str(t)])), t) for t in ts), reverse=True)[:2]
    score[m] = dict(pairs=[t for _, t in d], d=[x for x, _ in d], score=float(np.mean([x for x, _ in d])))
top = sorted(score, key=lambda m: -score[m]["score"])[:4]
sel = dict(candidates=cand, scores=score, selected_models=top,
           pairs=[[m, t, fix[m]["est"][str(t)], k3[m]["est"][str(t)]] for m in top for t in score[m]["pairs"]])
json.dump(sel, open(os.path.join(HERE, "results", "t367_selection.json"), "w"), indent=1)
print("SELECTED:", json.dumps(sel["pairs"]))
