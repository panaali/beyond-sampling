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

### Amendment A (2026-10-19 09:40, after L2 seeds 0-4 and D31 seed 0, before D31 seeds 1-4 and any recipe run)

- **Disclosure:** D31 uses the scaled warm-up (2^15 steps, 1/64 of the run, the zoo's schedule shape), not the
  literal default 2^14; `train.py ... 31 scaled`.
- **Observation so far** [measured]: L2 accuracies 0.4866/0.6002/0.6026/0.4961/0.6018 (mean 0.5575) and D31 seed 0
  0.478 sit on the zoo's per-seed values (0.477/0.602/0.602/0.498/0.602). Together with the reproduction result
  (9/10 seeds converge to the zoo's own weights), the initial weights, not the amount or speed of training, decide
  the (4,4) basin. More data or lr x2 therefore does not test "undertraining" in the intended sense.
- **T111b (runs only if the pre-registered test ends "inconclusive about undertraining"):** search new seeds for
  better (4,4) basins. Seeds 5-44 at n_train = 2^28, scaled warm-up (tag `28s`, ~7 min each; screening only); the
  5 seeds with the highest 2^28 accuracy are retrained at the exact zoo setting (2^30, tag `30s`). Those 5 models
  are the T111b set; "better trained" again needs mean accuracy >= 0.606. The recipe test, truth, noisy copies and
  pass rule are exactly as for T111 (`run111.py 30s`). If no seed in 5-44 exceeds 0.62 at 2^28, report that
  H = 4, T = 4 shows no better basin in 45 seeds (the zoo-trained regime is the ceiling we can reach), and T111
  stays inconclusive on undertraining at this size.
- The recipe is still run on L2 and D31 as pre-registered (they are the "primary/secondary" sets of T111 proper).

## Status

T86 reproduction done (match); T111 done (FAIL; undertraining inconclusive at H=4); (7,3) and (4,7) training (2026-10-19 afternoon).

## Results

### T86 reproduction: (4,3) and (5,3) MATCH the zoo (pre-registered M1-M3) [measured]

Command: `bash queue2.sh jobs.txt` (2 workers) -> `python evaluate.py --no-zoo` -> `python match.py 30s`.
Accuracy on 2^23 N(0,1) samples (s.e. <= 1e-4); zoo rows in results.json from the same evaluate.py.

| size | our acc seeds 0-4 | zoo acc range | R | M1 | our n_store | zoo mode | M2 | our bnd1 median | M3 | match |
|---|---|---|---|---|---|---|---|---|---|---|
| (4,3) | 0.9860 0.8922 0.7594 0.9038 0.9449 | 0.7590-0.9846 | 0.646-1.097 | yes | 2/2 2/2 2/2 1/3 2/2 | 2/2 | yes | 0.213 | yes | **yes** |
| (5,3) | 0.9992 0.9988 0.9987 0.9970 0.9988 | 0.9420-0.9991 | 0.913-1.028 | yes | 2/3 x5 | 2/3 | yes | 1.000 | yes | **yes** |

Paired by seed (same initial weights as the zoo, bitwise; different data stream: CPU vs CUDA RNG). Reported, not
scored. **9 of 10 seeds land on essentially the zoo's own weights**: relative L2 distance of all weights
0.001-0.005, max |dw| 0.014-0.096, accuracy difference <= 0.0015, same n_store. The exception is (5,3) seed 0:
ours reaches 0.9992 with the 2/3 storage pattern, the zoo's 0.9420 with 1/4 (rel. weight distance 1.17, a different
basin). So which basin a seed falls into is mostly fixed by the initialisation; the data stream decides it only
occasionally (1 of 10 here).

| size | seed | zoo acc | our acc | diff | zoo n_store | our n_store | rel. weight distance |
|---|---|---|---|---|---|---|---|
| (4,3) | 0 | 0.9846 | 0.9860 | +0.0015 | 2/2 | 2/2 | 0.001 |
| (4,3) | 1 | 0.8928 | 0.8922 | -0.0006 | 2/2 | 2/2 | 0.005 |
| (4,3) | 2 | 0.7590 | 0.7594 | +0.0004 | 2/2 | 2/2 | 0.003 |
| (4,3) | 3 | 0.9043 | 0.9038 | -0.0005 | 1/3 | 1/3 | 0.002 |
| (4,3) | 4 | 0.9452 | 0.9449 | -0.0003 | 2/2 | 2/2 | 0.002 |
| (5,3) | 0 | 0.9420 | 0.9992 | +0.0573 | 1/4 | 2/3 | 1.171 |
| (5,3) | 1 | 0.9991 | 0.9988 | -0.0003 | 2/3 | 2/3 | 0.003 |
| (5,3) | 2 | 0.9987 | 0.9987 | -0.0000 | 2/3 | 2/3 | 0.001 |
| (5,3) | 3 | 0.9968 | 0.9970 | +0.0002 | 2/3 | 2/3 | 0.002 |
| (5,3) | 4 | 0.9987 | 0.9988 | +0.0001 | 2/3 | 2/3 | 0.002 |

Wall time 1360-1690 s per model (one thread, 2 jobs in parallel).

New rungs (7,3), (4,7): training after T111b.

### T111 proper: T87's recipe (unchanged) on lr x2 and 2x-data (4,4) models. FAILS; inconclusive on undertraining [measured]

Commands: `python run111.py 30sL2 8`, `python run111.py 31s 8` -> run111_<tag>.json/.log. Truth: estT exact,
matching 1e7-sample MC (|z| <= 1.0 on all 10 models). Sampling at 96 ops per sample.

| set | seed | acc (exact) | n_store | recipe err | recipe ops | clean ratio | noisy acc | noisy RMSE | noisy ratio |
|---|---|---|---|---|---|---|---|---|---|
| L2 (primary) | 0 | 0.4866 | 1/3 | -0.030 pp | 48.5M | **0.18** | 0.409 | 0.071% | **0.91** |
| L2 | 1 | 0.6004 | 2/2 | -0.712 pp | 37.9M | 83 | 0.439 | 0.290% | 13 |
| L2 | 2 | 0.6029 | 2/2 | -0.750 pp | 38.1M | 93 | 0.442 | 0.271% | 13 |
| L2 | 3 | 0.4965 | 1/3 | -0.231 pp | 43.3M | 9.6 | 0.433 | 0.840% | 131 |
| L2 | 4 | 0.6021 | 2/2 | -0.518 pp | 39.1M | 46 | 0.513 | 0.280% | 14 |
| D31 (secondary) | 0 | 0.4776 | 2/2 | +0.070 pp | 55.7M | 1.14 | 0.346 | 0.419% | 33 |
| D31 | 1 | 0.5996 | 2/2 | -0.556 pp | 36.7M | 49 | 0.423 | 0.336% | 18 |
| D31 | 2 | 0.6001 | 2/2 | -0.579 pp | 36.5M | 53 | 0.425 | 0.308% | 17 |
| D31 | 3 | 0.4988 | 1/3 | -0.202 pp | 43.0M | 7.3 | 0.430 | 0.616% | 69 |
| D31 | 4 | 0.6006 | 2/2 | -0.594 pp | 36.3M | 56 | 0.492 | 0.300% | 15 |

Zoo (T87, same recipe): clean 459/71/76/7.9/76, noisy 91/33/14/136/17.

- Mean accuracy L2 0.5575, D31 0.5553 (zoo 0.556): **neither is "better trained"** (bar 0.606). L2 is the
  primary set (higher mean). By the pre-registered rule the T111 test is **inconclusive about undertraining**;
  the recipe fails the rung on both sets (4/5 and 5/5 seeds lose clean, 4/5 and 5/5 lose noisy).
- Both variants land on the zoo's per-seed basins (accuracy within 0.01 of the zoo seed, same n_store), except
  L2 seed 0 (1/3 storage, acc 0.487 vs zoo 0.477 with 2/2). That one different basin is the only model where the
  recipe wins (clean 0.18, noisy 0.91), against 459/91 for zoo seed 0. [measured, n = 1; suggests the recipe's
  failure depends on the basin, not on T alone; speculative until T111b]
- The recipe's errors on the 0.60-basin models (seeds 1, 2, 4 in every set) are about -0.5 to -0.75 pp, the same
  as on the zoo, i.e. the failure reproduces on independently trained copies of that basin.
- Amendment A therefore applies: T111b (seed search) below.

### T111b: seed search for better (4,4) basins (amendment A). No better basin; recipe FAILS [measured]

Screening (`queue2.sh jobs111b.txt`, seeds 5-44, n_train 2^28 scaled warm-up): accuracies 0.408-0.601, best
seed 42 0.6012, then 20 0.5928, 40 0.5748, 10 0.5604, 44 0.5599; **no seed exceeds 0.62**. Two basins recur:
the "0.60" basin (2/2 storage) and lower 1/3 or 2/2 basins at 0.41-0.56.
Top 5 retrained at the zoo setting (2^30, `jobs_phase3.txt`), recipe `python run111.py 30s 8 42,20,40,10,44`:

| seed | acc (exact) | n_store | recipe err | recipe ops | clean ratio | noisy acc | noisy RMSE | noisy ratio |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.6017 | 2/2 | -0.691 pp | 39.3M | 82 | 0.474 | 0.213% | 7.7 |
| 20 | 0.5944 | 2/2 | +0.388 pp | 34.8M | 23 | 0.501 | 0.462% | 30 |
| 40 | 0.5611 | 2/2 | +0.069 pp | 28.2M | **0.56** | 0.334 | 0.107% | 1.06 |
| 10 | 0.5575 | 2/2 | +0.105 pp | 23.6M | 1.10 | 0.362 | 0.332% | 15 |
| 44 | 0.5570 | 1/3 | +0.113 pp | 35.9M | 1.94 | 0.376 | 0.335% | 15 |

Mean accuracy 0.574 (< 0.606: not better trained). The recipe loses on 4/5 seeds clean and 5/5 noisy.
**Truth caveat:** estT exact reads ~2e-4 LOW on seed 40 (1e7 MC z +3.4; independent 5e7 MC, rng 9040, z +2.9);
z is positive on 4 of these 5 models (+1.0 to +1.8 elsewhere). The estT exact mode is therefore accurate only to
~1e-4-2e-4 here (cheap orthant quadrature on thin simplices is the likely cause; T87 diagnosis item 2). This
changes no verdict: on MC truth seed 40's clean error is ~+0.03-0.05 pp (still < sampling's 0.092%), and every
failing seed fails by >= 0.1 pp or on its noisy copies (whose truth is MC). Note seed 10 changed basin between
2^28 (1/3) and 2^30 (2/2).

### T111 verdict (pre-registered rule + amendment A)

- **Rung (4,4) FAILS for T87's recipe on every better-training attempt**: lr x2 (4/5 seeds lose), 2x data
  (5/5), best of 45 seeds (4/5 clean, 5/5 noisy). 20 models in all, 3 pass clean, 0 pass both clean and noisy except
  L2 seed 0 (0.18 / 0.91).
- **Undertraining could not be tested at H = 4, T = 4**: no training change we tried (lr x2, 2x data, 40 new
  seeds) moves the models out of the bulk-error regime (best accuracy 0.6029; bnd1 0.10-0.14, i.e. errors are not
  at the boundaries). The zoo's (4,4) models are not undertrained in the sense of "more training would fix them":
  0.60 looks like what this architecture reaches at this size with this recipe [inferred from 45 seeds + 2 variants].
- So the question "is T87's failure about T or about undertraining" is **inconclusive at H = 4**; the clean way to
  separate them is a T step in a well-trained regime, e.g. (8,3) -> (8,4) (zoo (8,4) is 0.97-0.99 accurate), which
  is the ladder's main path (TASKS ladder rule 5) and is T192's domain (recipe108 passes (8,4), fails (8,5)).
- [speculative] The recipe's error depends on the basin more than on accuracy: the three clean wins (L2 0, 30s 40,
  31s 0 nearly) are not the most accurate models, and the 0.60 basin (2/2 storage with all-four-on bulk errors,
  T87 item 3) fails in every copy (-0.5 to -0.75 pp).


## Compute accounting

Training: CPU wall time per model in each npz (`wall_seconds`), single thread. Evaluation: 2^23 forward passes per
model. T111: estT ops (its own counter, `m43.Ops`) for exact and recipe; 1e7-sample MC per noisy copy is truth only.

## Claims

- [measured] alg_zoo's training code, unchanged, at the zoo's settings on CPU reproduces the zoo's (4,3) and (5,3)
  models: all pre-registered match tests pass, and 9/10 seeds converge to within 0.5% (rel. L2) of the zoo weights.
- [inferred] Models we train at missing sizes with the same code are zoo-like models (same pipeline, same
  per-seed basins), not a different training regime.

## Alternatives not ruled out

- Our data stream uses the CPU RNG (zoo: CUDA), so per-seed outcomes differ; only seed distributions are comparable.
- PyTorch 2.14.1+cpu here vs 2.9.1+cu128 for the zoo; float32 nondeterminism differs.

## Failure modes

- With n = 5 seeds per size the match test has little power; a bimodal size (e.g. zoo (4,3): 0.76-0.98) can match
  almost anything inside its wide range.
