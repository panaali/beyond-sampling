"""T234 step 0: train one (16,10) model with the UNCHANGED alg_zoo.training.train (all defaults: 2^30 sequences),
then evaluate it like every other model. usage: python repro_zoo.py seed -> results/zoo_code_seed{seed}.json"""
import json, os, sys, time
import numpy as np
import torch as th
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from evaluate import evaluate
seed = int(sys.argv[1])
out = os.path.join(HERE, "..", "..", "..", "data", "capacity_ceiling", f"zoo_code_seed{seed}"); os.makedirs(out, exist_ok=True)
os.environ["RESULTS_DIR"] = out; os.chdir(out)
from alg_zoo.training import train
t0 = time.time(); train("2nd_argmax", hidden_size=16, seq_len=10, seed=seed); wall = time.time() - t0
sd = th.load(os.path.join(out, f"2nd_argmax_16_10_{2 ** 30}_{seed}.pth"), map_location="cpu")
W = (sd["rnn.weight_ih_l0"][:, 0].double().numpy(), sd["rnn.weight_hh_l0"].double().numpy(), sd["linear.weight"].double().numpy())
np.savez(os.path.join(HERE, "weights", f"zoo_code_seed{seed}.npz"), Wi=W[0], Wh=W[1], Wo=W[2])
e = evaluate([W], 10 ** 8)[0]
json.dump(dict(seed=seed, wall_sec=wall, eval=e), open(os.path.join(HERE, "results", f"zoo_code_seed{seed}.json"), "w"), indent=1)
print("zoo code seed", seed, "acc %.5f +- %.5f wall %.0fs" % (e["acc"], e["se"], wall))
