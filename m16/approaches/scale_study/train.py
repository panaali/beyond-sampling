"""Train one grid model for the T421 scale study with AlgZoo's own training code (alg_zoo.training.train), unchanged.

Everything is the alg_zoo default used for the zoo (AdamW, weight decay 1e-3, lr 0.04/sqrt(H), 2^14 warm-up steps,
exponential decay to 0.01x by mid-run, batch 1024, n_train = 2^30 fresh N(0,1) samples, torch.manual_seed(seed), no bias),
except n_val = 2^20 (validation never touches the weights). CPU, one thread: this is the "cpu" seed stream (the zoo itself
was trained on GPU, so zoo seed s and our cpu seed s are different random draws of the same recipe).

usage: python train.py H T SEED
Writes data/scale_study/runs/<H>_<T>_<SEED>/ (alg_zoo checkpoints + jsonl logs) and, in this folder,
  weights/m_<H>_<T>_cpu<SEED>.pth   alg_zoo state_dict (load with alg_zoo.loading.load_dist_rnn)
  weights/m_<H>_<T>_cpu<SEED>.npz   Wi (H,), Wh (H,H), Wo (T,H) float64 (T86 ladder_training format)
  results/models/m_<H>_<T>_cpu<SEED>.json   accuracy (1e7 MC, evaluate.py), wall time, provenance
"""
import json
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch as th

th.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "../../../data/scale_study"))
sys.path.insert(0, HERE)
from common_io import save_model  # noqa: E402


def main(H, T, seed):
    run_dir = os.path.join(DATA, "runs", f"{H}_{T}_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.environ["RESULTS_DIR"] = run_dir
    from alg_zoo.training import train
    t0 = time.time()
    train("2nd_argmax", hidden_size=H, seq_len=T, seed=seed, n_val=2**20)
    wall = time.time() - t0
    pth = os.path.join(run_dir, f"2nd_argmax_{H}_{T}_{2**30}_{seed}.pth")
    val = [json.loads(l) for l in open(os.path.join(run_dir, f"val_2nd_argmax_{H}_{T}_{2**30}_{seed}.jsonl"))]
    save_model(f"m_{H}_{T}_cpu{seed}", th.load(pth), H=H, T=T, seed=seed, stream="cpu",
               provenance={"source": "T421 train.py (alg_zoo.training.train, defaults except n_val=2^20)",
                           "torch": th.__version__, "wall_seconds": wall, "threads": 1,
                           "val_acc_2^20": float(np.mean([v["accuracy"] for v in val]))})
    shutil.copy(pth, os.path.join(run_dir, "final.pth"))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:4]))
