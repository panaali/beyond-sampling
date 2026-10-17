# T86: training the missing ladder sizes with AlgZoo's own training code

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [understand_ladder (T73/T93/T94)](../understand_ladder/README.md)

## Question

The ladder protocol (TASKS.md) climbs one step at a time in H (neurons) and T (sequence length), but the zoo only has
H in {2,3,4,5,6,8,16,32} and T in {2,3,4,5,6,8,10}. Can we train the missing 2nd-argmax RNNs (first H=7 at T=3, then
T=7 at H=4) on one CPU core with the zoo's own code and settings, and are they close enough to zoo models to sit on
the ladder?

## Method

**Zoo settings** (from `alg_zoo/training.py` defaults and the alg-zoo README, "trained using
`train(task_name, hidden_size=hidden_size, seq_len=seq_len, seed=seed)` with all other arguments set to their
default values"; PyTorch 2.9.1+cu128 on an A100):
- 1-layer ReLU RNN, no biases (`DistRNN`), readout `Linear(H, T)`; init = PyTorch default after `torch.manual_seed(seed)`, seeds 0-4.
- inputs **N(0,1)** (`th.randn`), not Unif[-1,1]; target `argsort(x)[-2]`; cross-entropy loss.
- AdamW, lr = 0.04/sqrt(H), weight decay 1e-3, batch 1024, n_train = 2^30 sequences (2^20 steps).
- lr multiplier = min(step/2^14, 1) · max(0.01^(2·step/n_steps), 0.01): linear warm-up over 2^14 steps, exponential
  decay reaching 0.01x at mid-run, then flat.
- The zoo's best-of-5 seed is chosen by accuracy; all 5 seeds are published.

**What we ran.** `train.py` calls `alg_zoo.training.train` unchanged. We checked that our initial weights are
bitwise identical to the zoo's `n_seqs=0` checkpoints for M_4,3 and M_5,3 seeds 0-4 [measured]. The data stream
can not be identical (zoo used the CUDA RNG, we use the CPU RNG), so per-seed outcomes may differ.

**Compromise: n_train = 2^28 with the warm-up scaled to keep the schedule shape.** On this shared machine one step
costs 2.7-6 ms on one core (A100: 1.0-1.5 ms per step in the zoo logs), so 2^30 sequences takes 50-100 min per T=3 model.
We used n_train = 2^28 (2^18 steps, 12-27 min per model). With fewer steps the default 2^14-step warm-up becomes a
large fraction of the run and, because the decay reaches its floor at mid-run, the peak lr is never reached
(at 2^25 the lr peaks at 0.0016 instead of 0.02). We therefore set `lr_warmup_steps = 2^14 · n_train / 2^30`
(warm-up = 1/64 of the run, as in the zoo). This is the only argument besides n_train (and n_val, which never
touches the weights) that differs from the zoo defaults. Files and data use the tag `28s` (= 2^28, scaled warm-up).

**Evaluation** (`evaluate.py`, numpy float64): accuracy on 2^23 = 8.4M N(0,1) samples (fixed rng seed 12345;
s.e. ≤ 1e-4), plus mechanism signatures chosen before looking at our models:
- `t1`: distance |t_k − 1| of the step-2 switch nearest to x1 = x0 (worst of the two charts sign(x0)), with
  t_k = −(Wh diag(g1) Wi)_k / Wi_k as in understand_ladder step 2;
- `n_store`: how many neurons hold |x0| after step 1 in each chart;
- `bnd1`: share of misclassified samples within 1° of a relevant true boundary (2nd-largest equal to largest or third-largest).

Files: [train.py](train.py) (one model), [queue.sh](queue.sh) (sequential runs on one core),
[evaluate.py](evaluate.py) (→ [results.json](results.json)), [report.py](report.py) (tables), [models.py](models.py) (loader).
Weights: `data/ladder_training/m_<H>_<T>_<tag>_<seed>.npz` (Wi (H,), Wh (H,H), Wo (T,H), wall time);
the alg_zoo checkpoints and jsonl training logs are in `data/ladder_training/runs/`; the zoo's own training logs
(downloaded from `gs://arc-ml-public/alg/zoo/train_*.jsonl`) are in `data/ladder_training/zoo_logs/`.

```python
import sys; sys.path.insert(0, "m16/approaches/ladder_training")
from models import load, accuracy
Wi, Wh, Wo = load(7, 3, seed)      # seeds 0-4; also load(4, 7, seed)
```

## Status

RESULTS_PLACEHOLDER
