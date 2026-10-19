"""Re-save a finished run whose save step failed (T421: the first models of the helper containers finished before
weights/ existed). Reads the final alg_zoo checkpoint in data/scale_study/runs/<H>_<T>_<SEED>/ and registers it exactly
as train.py would; wall time taken from the last train log line. usage: python recover.py H T SEED"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch as th

from common_io import save_model
from train import DATA

H, T, seed = map(int, sys.argv[1:4])
run_dir = os.path.join(DATA, "runs", f"{H}_{T}_{seed}")
stem = f"2nd_argmax_{H}_{T}_{2**30}_{seed}"
train_log = [json.loads(l) for l in open(os.path.join(run_dir, f"train_{stem}.jsonl"))]
assert train_log[-1]["step"] == 2**20, "training did not finish"
val = [json.loads(l) for l in open(os.path.join(run_dir, f"val_{stem}.jsonl"))]
save_model(f"m_{H}_{T}_cpu{seed}", th.load(os.path.join(run_dir, f"{stem}.pth")), H=H, T=T, seed=seed, stream="cpu",
           provenance={"source": "T421 train.py (alg_zoo.training.train, defaults except n_val=2^20); re-saved by recover.py",
                       "torch": th.__version__, "wall_seconds": train_log[-1]["seconds"], "threads": 1,
                       "val_acc_2^20": float(np.mean([v["accuracy"] for v in val]))})
