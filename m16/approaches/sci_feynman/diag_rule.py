"""Diagnosis: spread of the rule's error under random shifts (iid vs sobol), original model.
usage: python diag_rule.py budget reps design..."""
import sys, json, time
import numpy as np
import feyn as F, wtool as W
budget = int(sys.argv[1]); reps = int(sys.argv[2]); designs = sys.argv[3:]
M = W.get_model("orig"); truth = json.load(open("/home/user/clean-neurons/results/leaderboard_truths.json"))["orig"]["acc"]
for d in designs:
    E, Cs, parts = [], [], []
    t0 = time.time()
    for r in range(reps):
        e, c, p = F.estimate(M, budget, d, shift_seed=r); E.append(e); Cs.append(c); parts.append([p["r23"], p["r12"], p["blind"]])
    E = np.array(E); parts = np.array(parts); C_ = np.mean(Cs)
    samp = truth * (1 - truth) / C_
    print(f"{d:6s} budget {budget} cost {C_:.0f} n={p['n']} mean {E.mean():.5f} (truth {truth:.5f}) bias {100*(E.mean()-truth):+.3f}+-{100*E.std()/np.sqrt(reps):.3f}pp "
          f"sd {100*E.std():.3f}pp  MSE/sampling {np.mean((E-truth)**2)/samp:.2f}  part sds(pp) {np.round(100*parts.std(0),3)}  {time.time()-t0:.0f}s", flush=True)
