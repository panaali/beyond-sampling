# T419: which other tasks does the exact M_16,10 architecture learn to ~95%?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [OWNER_QA 6.6](../../../OWNER_QA.md#q6-6) · [T86 training pipeline](https://github.com/panaali/clean-neurons/tree/claude/t86-ladder-training/m16/approaches/ladder_training)

## Question

The owner asked (OWNER_QA 6.6) which of our positive estimation methods transfer to the same architecture trained on a
different task. This lane finds about 5 tasks that the exact M_16,10 architecture (16 ReLU units, no bias, 10 N(0,1)
inputs, 10-way readout of the final state, the zoo's training recipe and schedule) learns to >= 90-95% accuracy,
at least 3 of them not "k-th largest". Later sessions test method transfer on the picked tasks.

## Method

- **Tasks**: [tasks.py](tasks.py) (torch and numpy; `labels(name, x)`; `register()` adds them to `alg_zoo`'s
  `task_registry` as continuous/dist tasks). Every label is a.s. unique (ties have probability 0).
- **Training**: [train.py](train.py) calls `alg_zoo.training.train` **unchanged** (as T86's `train.py`, whose pipeline
  reproduces zoo weights): DistRNN H = 16, T = 10, no bias, PyTorch default init after `torch.manual_seed(seed)`,
  N(0,1) inputs, cross-entropy, AdamW lr 0.04/sqrt(16) = 0.01, weight decay 1e-3, batch 1024, lr multiplier
  min(step/warm, 1) * max(0.01^(2 step/n_steps), 0.01). Zoo schedule: n_train = 2^30 (2^20 steps), warm = 2^14.
  Only n_val = 2^20 differs (validation never touches the weights). CPU, 1 thread per process, `ulimit -v 2500000`.
- **Accuracy**: [evaluate.py](evaluate.py), numpy float64 forward pass on the same 10^7 N(0,1) samples for every
  model (rng seed 419), s.e. <= 1e-4.

## Pre-registration (written 2026-10-19 before any training)

### Candidate tasks (17)

Positions are 0-based. "Pair" tasks score i = 1..9 on (x_{i-1}, x_i), so class 0 is never the target.

| # | name | label | family | why |
|---|---|---|---|---|
| 1 | `argmax` | position of the largest | k-th largest (k = 1) | easiest rank task |
| 2 | `2nd_argmax` | position of the 2nd largest | k-th largest (k = 2) | **control**: M_16,10's task; must land near the zoo's .953 |
| 3 | `3rd_argmax` | 3rd largest | k-th largest | owner list |
| 4 | `4th_argmax` | 4th largest | k-th largest | |
| 5 | `5th_argmax` | 5th largest (= `alg_zoo` argmedian at T = 10) | k-th largest | |
| 6 | `2nd_argmin` | 2nd smallest | k-th largest (on -x) | **control**: same task as #2 on -x (N(0,1) is symmetric); owner list |
| 7 | `argmax_absdiff` | argmax_i \|x_i - x_{i-1}\| | adjacent pair | owner list |
| 8 | `argmin_absdiff` | argmin_i \|x_i - x_{i-1}\| (closest adjacent pair) | adjacent pair | |
| 9 | `argmax_diff` | argmax_i x_i - x_{i-1} (largest upward jump) | adjacent pair | |
| 10 | `argmax_pairsum` | argmax_i x_i + x_{i-1} | adjacent pair | owner list |
| 11 | `argmax_window3` | argmax_i x_{i-2} + x_{i-1} + x_i, i = 2..9 | adjacent window | longer memory than a pair |
| 12 | `2nd_argmax_abs` | position of the 2nd largest \|x_i\| | rank of \|x\| | owner list |
| 13 | `argmax_abs` | largest \|x_i\| | rank of \|x\| | |
| 14 | `argmin_abs` | value closest to 0 | rank of \|x\| | |
| 15 | `argmax_cumsum` | argmax_i (x_0 + ... + x_i) (where the running sum peaks) | running sum | carries a sum and its running max |
| 16 | `argoutlier` | argmax_i \|x_i - mean(x)\| (`alg_zoo`'s own task) | global statistic | needs the mean, known only at the end |
| 17 | `n_records` | number of left-to-right maxima minus 1 (0..9) | count | a count, not a position; tests a counter in the state |

"Rank tasks" below = #1-6 and #12-14 (a rank of a pointwise function of x_i). "Non-rank" = #7-11 and #15-17
(each label depends on combinations of several inputs).

### Stage A: pre-screen (cheap, not used for final accuracies)
All 17 tasks, seed 0, n_train = 2^26 with the zoo schedule shape (warm-up 2^10 steps = 1/64 of the run), ~3 min each.
Used only to choose stage-B tasks:
- tasks with stage-A accuracy < 0.70 are dropped (reported as failures; the zoo schedule gives 16x more data, so a
  task at 0.70 could still reach ~0.9; the control `2nd_argmax` at 2^26 calibrates how much 2^26 understates 2^30);
- tasks with stage-A accuracy >= 0.995 get one zoo-schedule seed only (too easy to sit in the .90-.95 band; reported);
- if more than 12 tasks remain, keep the 12 with the highest stage-A accuracy, always keeping `2nd_argmax`.

### Stage B: zoo schedule (final accuracies)
Each stage-B task: seeds 0, 1, 2 at n_train = 2^30, warm-up 2^14 (exactly the zoo setting). Accuracy = 10^7-sample MC.
Compute: ~45 min per model on one core. Runs are split over this container (2 workers) and up to 2 child sessions
(2 workers each, own containers), as allowed by the standing rules.

### Pick rule (5 tasks)
1. Eligible: stage-B mean accuracy >= 0.90. Controls `2nd_argmax` and `2nd_argmin` are not eligible (they are
   M_16,10's own task).
2. Order eligible tasks by |mean - 0.95| (closest first).
3. Take them greedily in that order subject to: at most 2 from the k-th largest family (#1-5), at most 1 from the
   rank-of-|x| family (#12-14), at most 2 from the adjacent pair/window family (#7-11), and at least 3 non-rank
   picks overall (if the greedy pass would end with fewer than 3, the remaining slots go to the closest non-rank tasks).
4. If fewer than 5 tasks are eligible, relax the bar to 0.85 for the remaining slots and flag them.
- Each pick's **primary model** is its most accurate seed (the zoo's convention); the other seeds are retrains.
- Control check: `2nd_argmax` mean should lie within 0.02 of the zoo's 0.953 and `2nd_argmin` within 0.02 of
  `2nd_argmax`. If not, the pipeline is flagged before anyone uses the picks.

## Status

Pre-registered; training not started.

## Results

(pending)

## Compute accounting

Training: forward + backward passes on 1024-sequence batches (2^20 steps per zoo-schedule model). Evaluation:
10^7 float64 forward passes per model. Wall times are stored in each weight file.

## Claims

(pending)

## Alternatives not ruled out

- 3 seeds per task: a task can have bad basins (the zoo's own 2nd-argmax seeds vary widely at small H). A task below
  0.90 here might reach it on other seeds.
- CPU vs the zoo's CUDA RNG: same initial weights, different data stream (T86 found this changes per-seed outcomes
  rarely at small sizes).

## Failure modes

- Tasks with class 0 unused (pair tasks) still have a 10-way head; the follow-ups must use the same label convention.
