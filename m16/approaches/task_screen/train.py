"""Train one T419 model with alg_zoo.training.train UNCHANGED (zoo defaults), only the task differs.

usage: python train.py TASK seed log2_n_train
  H = 16, T = 10, DistRNN without bias, AdamW, lr 0.04/sqrt(16), wd 1e-3, batch 1024, decay to 0.01x by mid-run.
  Warm-up = 2^14 * n_train / 2^30 steps (the zoo's schedule shape; at log2_n_train = 30 it IS the zoo default 2^14).
  n_val = 2^20 (validation only, never touches the weights).
Writes data/task_screen/runs/<TASK>_<log2n>_<seed>/ (alg_zoo .pth checkpoints + jsonl logs) and
data/task_screen/<TASK>_<log2n>_<seed>.npz with Wi (16,), Wh (16,16), Wo (10,16) float64 + wall time,
and the final state dict (zoo format) as data/task_screen/<TASK>_<log2n>_<seed>.pth.
"""
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch as th

th.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/task_screen"))
H, T = 16, 10


def main(task, seed, log2n):
    tasks.register()
    name = "t419_" + task
    run_dir = os.path.join(ROOT, "runs", f"{task}_{log2n}_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.environ["RESULTS_DIR"] = run_dir
    from alg_zoo.training import train
    warm = 2**14 * 2**log2n // 2**30
    t0 = time.time()
    train(name, hidden_size=H, seq_len=T, seed=seed, n_train=2**log2n, n_val=2**20, lr_warmup_steps=warm)
    wall = time.time() - t0
    pth = os.path.join(run_dir, f"{name}_{H}_{T}_{2**log2n}_{seed}.pth")
    sd = th.load(pth)
    out = os.path.join(ROOT, f"{task}_{log2n}_{seed}")
    shutil.copy(pth, out + ".pth")
    np.savez(out + ".npz", Wi=sd["rnn.weight_ih_l0"].numpy()[:, 0].astype(np.float64),
             Wh=sd["rnn.weight_hh_l0"].numpy().astype(np.float64), Wo=sd["linear.weight"].numpy().astype(np.float64),
             wall_seconds=wall, torch_version=th.__version__, n_train=2**log2n, lr_warmup_steps=warm, task=task)
    print(f"done {task} seed={seed} n=2^{log2n} wall={wall:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
