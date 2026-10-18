"""T234 add-on: train one (16,T) 2nd-argmax model with the UNCHANGED alg_zoo.training.train (defaults, 2^30).
usage: python train_zoo.py T seed -> weights/zoo_code_H16T{T}_seed{seed}.npz"""
import os, sys, time
import numpy as np
import torch as th
HERE = os.path.dirname(os.path.abspath(__file__))
T, seed = int(sys.argv[1]), int(sys.argv[2])
out = os.path.join(HERE, "..", "..", "..", "data", "capacity_ceiling", f"zoo_code_H16T{T}_seed{seed}"); os.makedirs(out, exist_ok=True)
os.environ["RESULTS_DIR"] = out; os.chdir(out)
from alg_zoo.training import train
t0 = time.time(); train("2nd_argmax", hidden_size=16, seq_len=T, seed=seed)
sd = th.load(os.path.join(out, f"2nd_argmax_16_{T}_{2 ** 30}_{seed}.pth"), map_location="cpu")
os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
np.savez(os.path.join(HERE, "weights", f"zoo_code_H16T{T}_seed{seed}.npz"), Wi=sd["rnn.weight_ih_l0"][:, 0].double().numpy(),
         Wh=sd["rnn.weight_hh_l0"].double().numpy(), Wo=sd["linear.weight"].double().numpy(), wall=time.time() - t0)
print("done", T, seed, time.time() - t0)
