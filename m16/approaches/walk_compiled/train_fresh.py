"""Train one fresh H=16, T=5 2nd-argmax RNN with alg_zoo/training.py at the zoo recipe (all defaults: n_train 2^30,
batch 1024, AdamW lr 0.04/sqrt(H), wd 1e-3, no bias). Seeds are pre-registered in README.md.
usage: python train_fresh.py SEED   (writes data/t393_models/2nd_argmax_16_5_*_SEED.pth; run from repo root)"""
import os
import sys

import torch as th

th.set_num_threads(1)
seed = int(sys.argv[1])
os.environ["RESULTS_DIR"] = os.path.abspath("data/t393_models")
os.makedirs(os.environ["RESULTS_DIR"], exist_ok=True)
from alg_zoo.training import train  # noqa: E402

train("2nd_argmax", hidden_size=16, seq_len=5, seed=seed, log_every=16384)
