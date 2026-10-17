"""Train one 2nd-argmax RNN with AlgZoo's own training code (alg_zoo.training.train), unchanged.

Only n_train (and n_val, which never touches the weights) differ from the zoo defaults; everything else
(AdamW, lr 0.04/sqrt(H), weight decay 1e-3, 2^14 warm-up steps, decay to 0.01x by mid-run, batch 1024,
N(0,1) inputs, torch.manual_seed(seed), no bias) is the alg_zoo default used for the zoo.

usage: python train.py H T seed log2_n_train [warmup]
  warmup = "default": lr_warmup_steps = 2^14 (zoo default; with small n_train the warm-up then eats most of the run
                      and the lr never gets near its peak, because decay already reaches 0.01x at mid-run)
  warmup = "scaled" : lr_warmup_steps = 2^14 * n_train / 2^30, i.e. the zoo's schedule *shape* (warm-up = 1/64 of run)
Writes data/ladder_training/runs/<H>_<T>_<log2n>[s]_<seed>/ (alg_zoo .pth checkpoints + jsonl logs) and
data/ladder_training/m_<H>_<T>_<log2n>[s]_<seed>.npz ("s" suffix = scaled warm-up) with Wi (H,), Wh (H,H), Wo (T,H) float64, plus wall time.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch as th

th.set_num_threads(1)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/ladder_training"))


def main(H, T, seed, log2n, warmup="default", n_val=2**20):
    tag = f"{log2n}s" if warmup == "scaled" else f"{log2n}"
    warm = 2**14 * 2**log2n // 2**30 if warmup == "scaled" else 2**14
    run_dir = os.path.join(ROOT, "runs", f"{H}_{T}_{tag}_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.environ["RESULTS_DIR"] = run_dir
    from alg_zoo.training import train  # imported after RESULTS_DIR is set (save_model reads it at call time anyway)
    t0 = time.time()
    train("2nd_argmax", hidden_size=H, seq_len=T, seed=seed, n_train=2**log2n, n_val=n_val, lr_warmup_steps=warm)
    wall = time.time() - t0
    sd = th.load(os.path.join(run_dir, f"2nd_argmax_{H}_{T}_{2**log2n}_{seed}.pth"))
    Wi = sd["rnn.weight_ih_l0"].numpy()[:, 0].astype(np.float64)
    Wh = sd["rnn.weight_hh_l0"].numpy().astype(np.float64)
    Wo = sd["linear.weight"].numpy().astype(np.float64)
    np.savez(os.path.join(ROOT, f"m_{H}_{T}_{tag}_{seed}.npz"), Wi=Wi, Wh=Wh, Wo=Wo, wall_seconds=wall,
             torch_version=th.__version__, n_train=2**log2n, lr_warmup_steps=warm)
    print(f"done H={H} T={T} seed={seed} n=2^{log2n} warmup={warm} wall={wall:.0f}s")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:5]), *sys.argv[5:6])
