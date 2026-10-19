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

**Earlier session (2026-10-17, superseded).** The first T86 session trained at n_train = 2^25-2^28 with the warm-up
scaled to keep the schedule shape (tags `25s`, `26s`, `28s`), because it shared 4 cores with many agents (2.7-6 ms
per step). Those rows are kept in results.json as history, but their weight files were never published and are not
in the recovered workspace (TASKS T94). **This session (2026-10-19) has its own container: one step costs ~2 ms on one
core, so a 2^30 model takes ~35-45 min and every model below is trained at the exact zoo setting** (tag `30s`; at
2^30 the scaled warm-up equals the zoo's 2^14 steps). Only n_val (validation size, never touches the weights) differs.
Weights are committed in [weights/](weights/) (a few hundred floats each) so other lanes can use them.

**Evaluation** (`evaluate.py`, numpy float64): accuracy on 2^23 = 8.4M N(0,1) samples (fixed rng seed 12345;
s.e. ≤ 1e-4), plus mechanism signatures chosen before looking at our models:
- `t1`: distance |t_k − 1| of the step-2 switch nearest to x1 = x0 (worst of the two charts sign(x0)), with
  t_k = −(Wh diag(g1) Wi)_k / Wi_k as in understand_ladder step 2;
- `n_store`: how many neurons hold |x0| after step 1 in each chart;
- `bnd1`: share of misclassified samples within 1° of a relevant true boundary (2nd-largest equal to largest or third-largest).

Files: [train.py](train.py) (one model), [queue.sh](queue.sh) (sequential runs on one core), [queue2.sh](queue2.sh) + [jobs.txt](jobs.txt) (this session: 2 workers),
[evaluate.py](evaluate.py) (→ [results.json](results.json)), [report.py](report.py) (tables), [models.py](models.py) (loader).
Weights: `data/ladder_training/m_<H>_<T>_<tag>_<seed>.npz`, copied to [weights/](weights/) when done, (Wi (H,), Wh (H,H), Wo (T,H), wall time);
the alg_zoo checkpoints and jsonl training logs are in `data/ladder_training/runs/`; the zoo's own training logs
(downloaded from `gs://arc-ml-public/alg/zoo/train_*.jsonl`) are in `data/ladder_training/zoo_logs/`.

```python
import sys; sys.path.insert(0, "m16/approaches/ladder_training")
from models import load, accuracy
Wi, Wh, Wo = load(7, 3, seed)      # seeds 0-4; also load(4, 7, seed)
```

## Pre-registration (T86 + T111, written 2026-10-19 before any training in this session)

### T86 jobs (alg_zoo.training.train unchanged, zoo settings, n_train = 2^30, seeds 0-4)
1. Reproduction: (H,T) = (4,3) and (5,3).
2. New rungs: (7,3) (H step between zoo 6 and 8) and (4,7) (T step between zoo 6 and 8).

### T86 match criterion (fixed now)
For a reproduced size, let the zoo's five seeds have accuracy range [a_min, a_max] and width w = a_max - a_min.
Widened range R = [a_min - max(w/2, 0.005), a_max + max(w/2, 0.005)].
- **(M1) accuracy:** all 5 of our seeds lie in R, and |median(ours) - median(zoo)| <= max(w, 0.01).
- **(M2) storage mechanism:** the modal `n_store` pattern of the zoo seeds occurs in >= 3 of our 5 seeds.
- **(M3) error geometry:** the median over seeds of `bnd1` (share of error mass within 1 degree of a true boundary)
  lies in the zoo's bnd1 range widened the same way (half-width, min 0.05).
- `t1` and `dead` are reported, not scored (t1 depends on the chart and is noisy across zoo seeds).
A size **matches** if M1, M2 and M3 all hold. If (4,3) or (5,3) fails, T86's new rungs are reported but flagged
"training not shown to reproduce the zoo" and the failure is diagnosed before the ladder uses them.

For a **new** size the same three tests are applied against the **union of the two zoo neighbours** (for (7,3):
zoo (6,3) and (8,3); for (4,7): zoo (4,6) and (4,8)), with R built from the pooled 10 seeds, and M2 replaced by:
n_store per chart lies between the neighbours' values (inclusive). This only checks that the new models
interpolate their neighbours; it can fail (e.g. a (7,3) seed at 0.95 or a (4,7) seed at 0.60 would fail M1).

### T111 jobs and test
Better-trained (4,4) models, seeds 0-4, two variants fixed now:
- **L2:** lr x2 (lr_base 0.08, T234's best variant), n_train = 2^30, tag `30sL2`;
- **D31:** zoo lr, n_train = 2^31 (2x data; 2^30 -> 2^31 was T234's largest gain), tag `31s`.
Zoo (4,4) 2^30 reference: accuracy 0.477-0.602, mean 0.556.
- **Primary set** = the variant with the higher mean accuracy (selection on accuracy only, before running the recipe).
  The recipe is also run on the other variant and reported (secondary).
- **"Better trained"** requires the primary set's mean accuracy >= 0.606 (zoo mean + 5 pp). If neither variant
  reaches it, T111 is reported as **inconclusive about undertraining** (training could not move these models out of
  the bulk-error regime), and the recipe result is still reported.
- **Recipe:** T87's `understand_m43/estT.py` imported UNCHANGED (mode "recipe", eps = 0.1), exactly as `run87.py`:
  clean truth = estT exact mode checked against 1e7-sample MC; 8 noisy copies per seed (absolute sigma 0.05, noise
  seed 50, `zoo43.noisy_copies`) with 1e7-sample MC truth; sampling cost 96 ops per sample (`ops_per_sample(4,4)`);
  MSE ratio = recipe squared error / (p(1-p) / (recipe ops / 96)); noisy ratio subtracts the MC-truth variance as
  run87 does. Script: [run111.py](run111.py) (a copy of run87.py's loop pointed at our weights).
- **Pass** (ladder protocol rule 2): clean MSE ratio < 1 on all 5 seeds AND noisy-copy ratio < 1 on all 5 seeds.
- **Reading:** pass on better-trained models -> T87's failure was about undertraining (bulk errors), not T.
  Fail on better-trained models -> the failure persists with better training; it is about T (or about the recipe's
  intermediate-gate freezing, T87 diagnosis item 1), at least up to the accuracy reached. Alternatives this cannot
  separate: (a) accuracy rises but errors stay bulk (check bnd1/bnd_med); (b) a pass caused only by higher accuracy
  shrinking p(1-p) is NOT a confound for the protocol but is reported (we report sampling RMSE next to the error).

## Status

Running (2026-10-19): training queue with 2 workers (`queue2.sh jobs.txt`).

## Results

Pending.

## Compute accounting

Training: CPU wall time per model in each npz (`wall_seconds`), single thread. Evaluation: 2^23 forward passes per
model. T111: estT ops (its own counter, `m43.Ops`) for exact and recipe; 1e7-sample MC per noisy copy is truth only.

## Claims

None yet.

## Alternatives not ruled out

- Our data stream uses the CPU RNG (zoo: CUDA), so per-seed outcomes differ; only seed distributions are comparable.
- PyTorch 2.14.1+cpu here vs 2.9.1+cu128 for the zoo; float32 nondeterminism differs.

## Failure modes

- With n = 5 seeds per size the match test has little power; a bimodal size (e.g. zoo (4,3): 0.76-0.98) can match
  almost anything inside its wide range.
