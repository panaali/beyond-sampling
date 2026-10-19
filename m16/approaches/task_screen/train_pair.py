"""Train one T419 PAIR-output model (amendment 1). Same architecture, init, data, optimizer and schedule as
alg_zoo.training.train (a line-for-line copy of its training loop); only the loss and the accuracy readout differ:
  loss    = softmax cross-entropy against the soft target with 1/2 on each of the two target positions
            (th.nn.functional.cross_entropy(logits, target_probs); the listwise top-2 loss)
  readout = the set of the two largest logits; accuracy = exact (unordered) set match.
usage: python train_pair.py TASK seed log2_n_train      (warm-up 2^14 * n_train / 2^30, as train.py)
Writes data/task_screen/<TASK>_<log2n>_<seed>.npz/.pth (same format as train.py) and runs/<...>/log.jsonl.
"""
import functools
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch as th

th.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import pair_soft_target  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/task_screen"))
H, T = 16, 10


def main(task, seed, log2n, weight_decay=0.001, lr_base=0.04, batch_size=1024, n_val=2**20):
    from alg_zoo.architectures import DistRNN
    from alg_zoo.training import lr_mult_schedule
    run_dir = os.path.join(ROOT, "runs", f"{task}_{log2n}_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    warm = 2**14 * 2**log2n // 2**30
    th.manual_seed(seed)
    model = DistRNN(hidden_size=H, seq_len=T, bias=False)
    opt = th.optim.AdamW(model.parameters(), lr=lr_base / H**0.5, weight_decay=weight_decay)
    n_steps = {"train": 2**log2n // batch_size, "val": n_val // batch_size}
    sched = th.optim.lr_scheduler.LambdaLR(opt, lr_lambda=functools.partial(
        lr_mult_schedule, n_steps=n_steps["train"], warmup_steps=warm, decay_min_mult=0.01))
    log = open(os.path.join(run_dir, "log.jsonl"), "w")
    t0 = time.time()
    for period in ["train", "val"]:
        sl = sa = 0.0
        with th.no_grad() if period != "train" else th.enable_grad():
            for step in range(n_steps[period]):
                x = th.randn(batch_size, T, dtype=model.dtype, device=model.device)
                tgt = pair_soft_target(task, x)
                logits = model.forward(x)
                loss = th.nn.functional.cross_entropy(logits, tgt)
                top2 = logits.topk(2, -1).indices
                acc = (tgt.gather(-1, top2).sum(-1) == 1.0).float().mean()
                if period == "train":
                    loss.backward()
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                sl += loss.item()
                sa += acc.item()
                if (step + 1) % 1024 == 0:
                    log.write(json.dumps({"period": period, "step": step + 1, "loss": sl / 1024, "accuracy": sa / 1024,
                                          "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}) + "\n")
                    log.flush()
                    sl = sa = 0.0
    wall = time.time() - t0
    sd = model.state_dict()
    out = os.path.join(ROOT, f"{task}_{log2n}_{seed}")
    th.save(sd, out + ".pth")
    np.savez(out + ".npz", Wi=sd["rnn.weight_ih_l0"].numpy()[:, 0].astype(np.float64),
             Wh=sd["rnn.weight_hh_l0"].numpy().astype(np.float64), Wo=sd["linear.weight"].numpy().astype(np.float64),
             wall_seconds=wall, torch_version=th.__version__, n_train=2**log2n, lr_warmup_steps=warm, task=task)
    print(f"done {task} seed={seed} n=2^{log2n} wall={wall:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
