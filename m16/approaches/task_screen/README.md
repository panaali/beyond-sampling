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

### Amendment 1 (2026-10-19 ~14:25 UTC, DRI messages relaying owner changes; written while stage A ran on tasks 1-17 only, before any stage-B (scored) training and before any amendment task trained)

The DRI relayed two owner changes: (i) pick **10** tasks, not 5, from a wider list of ~25-30; (ii) allow
**pair-output** tasks as a second tier. Stage A for tasks 1-17 had started under the original text; nothing else changes.

**New single-position tasks (7)** ([tasks.py](tasks.py)):

| # | name | label | family |
|---|---|---|---|
| 18 | `argmax_cumsum_abs` | argmax_i \|x_0 + ... + x_i\| | running sum |
| 19 | `argmax_suffixsum` | argmax_i (x_i + ... + x_9) (needs the running min of the prefix sum) | running sum |
| 20 | `argmax_absdiff_lag2` | argmax_i \|x_i - x_{i-2}\|, i = 2..9 | adjacent (lag 2) |
| 21 | `argmax_prod` | argmax_i x_i * x_{i-1} | adjacent pair (product) |
| 22 | `argmax_peak` | argmax_i x_i - (x_{i-1} + x_{i+1})/2, i = 1..8 (most prominent interior peak) | adjacent window |
| 23 | `n_sign_changes` | number of i with sign(x_i) != sign(x_{i-1}) (0..9) | count |
| 24 | `n_ascents` | number of i with x_i > x_{i-1} (0..9) | count |

**Pair-output tasks (5)**, label = an unordered pair of positions:

| # | name | label |
|---|---|---|
| 25 | `pair_top2` | positions of the two largest |
| 26 | `pair_extremes` | positions of the max and the min |
| 27 | `pair_top2_abs` | the two largest \|x_i\| |
| 28 | `pair_bottom2_abs` | the two values closest to 0 |
| 29 | `pair_closest` | the closest pair of values, argmin_{i<j} \|x_i - x_j\| |

Pair tasks keep the exact architecture (same 10 output units, no extra parameters). [train_pair.py](train_pair.py)
is a line-for-line copy of `alg_zoo.training.train`'s loop (same init after `torch.manual_seed(seed)`, same N(0,1)
batches, AdamW, lr, weight decay, warm-up and decay); only:
- **loss** = softmax cross-entropy against the soft target with probability 1/2 on each of the two target positions,
  `F.cross_entropy(logits, target_probs)` (a listwise top-2 loss);
- **readout** = the set of the two largest logits; **accuracy** = exact unordered set match. The decision boundary is
  the crossing of the 2nd- and 3rd-largest logits (the follow-ups must adapt the walk code to that).

**Stage A** runs on tasks 18-29 the same way (seed 0, 2^26). The 12-task cap on stage B is removed: every task with
stage-A accuracy >= 0.70 goes to stage B (3 seeds; 1 seed if stage A >= 0.995). Extra workers come from child sessions.

**Difference from M_16,10's algorithm** (fixed now, used to rank the picks):
D0 `2nd_argmin` (same task); D1 k-th largest (k != 2) and `pair_top2`; D2 rank of \|x\| (`argmax_abs`, `2nd_argmax_abs`,
`argmin_abs`) and `pair_extremes`, `pair_top2_abs`, `pair_bottom2_abs`; D3 adjacent/local (tasks 7-11, 20-22) and
`pair_closest`; D4 running sums, global statistics and counts (15-18, 19, 23, 24, `argoutlier`).

**Pick rule v2 (replaces the original pick rule):**
1. Eligible: stage-B mean accuracy >= 0.90 (pair tasks: set-match accuracy). `2nd_argmax` is never eligible;
   `2nd_argmin` only fills a slot left free at the end.
2. Hard caps among the 10: at most 3 pair tasks; at most 4 from the k-th largest family (`argmax`, `3rd`-`5th_argmax`,
   `2nd_argmin`), so at least 6 are not k-th largest.
3. Soft diversity caps: at most 2 rank-of-\|x\| single tasks, 3 adjacent/local single tasks, 2 running-sum tasks
   (15, 18, 19), 2 count tasks (17, 23, 24).
4. Order the eligible tasks by D (descending), then |mean - 0.95| (ascending); take them greedily under all caps.
   If fewer than 10 are taken, fill from the rest by |mean - 0.95| under the hard caps only; then `2nd_argmin` if a
   slot is still free.
5. The picks are reported ranked by D, then by |mean - 0.95|.
6. If fewer than 10 tasks are eligible, report how many are and the best near misses with their accuracies; the bar
   is not lowered. (The original rule 4, relaxing to 0.85, is withdrawn.)

## Status

Stage A done; stage B (zoo schedule) running in this container and 6 helper child sessions (W1-W6, branches claude/t419-w1..w6).

## Results

### Stage A (pre-screen; seed 0, n_train = 2^26, 10^6-sample MC, s.e. <= 5e-4)

| task | stage-A accuracy | decision |
|---|---|---|
| `argmax_suffixsum` | 0.9987 | stage B, 1 seed (>= 0.995) |
| `argmax` | 0.9964 | stage B, 1 seed (>= 0.995) |
| `argmax_peak` | 0.9963 | stage B, 1 seed (>= 0.995) |
| `argmax_diff` | 0.9939 | stage B, 3 seeds |
| `argmax_cumsum_abs` | 0.9938 | stage B, 3 seeds |
| `argmax_window3` | 0.9934 | stage B, 3 seeds |
| `argmax_absdiff` | 0.9926 | stage B, 3 seeds |
| `argmax_abs` | 0.9914 | stage B, 3 seeds |
| `argmax_pairsum` | 0.9904 | stage B, 3 seeds |
| `argmax_absdiff_lag2` | 0.9899 | stage B, 3 seeds |
| `argmin_absdiff` | 0.9896 | stage B, 3 seeds |
| `argmax_cumsum` | 0.9882 | stage B, 3 seeds |
| `argmin_abs` | 0.9872 | stage B, 3 seeds |
| `n_ascents` | 0.9597 | stage B, 3 seeds |
| `n_records` | 0.9576 | stage B, 3 seeds |
| `argmax_prod` | 0.9513 | stage B, 3 seeds |
| `n_sign_changes` | 0.9103 | stage B, 3 seeds |
| `argoutlier` | 0.9081 | stage B, 3 seeds |
| `pair_bottom2_abs` | 0.8816 | stage B, 3 seeds |
| `2nd_argmin` | 0.8700 | stage B, 3 seeds |
| `pair_top2` | 0.8536 | stage B, 3 seeds |
| `pair_top2_abs` | 0.8196 | stage B, 3 seeds |
| `pair_extremes` | 0.8153 | stage B, 3 seeds |
| `2nd_argmax` | 0.8048 | stage B, 3 seeds |
| `2nd_argmax_abs` | 0.6379 | drop (< 0.70) |
| `3rd_argmax` | 0.5061 | drop (< 0.70) |
| `4th_argmax` | 0.4698 | drop (< 0.70) |
| `5th_argmax` | 0.4450 | drop (< 0.70) |
| `pair_closest` | 0.1253 | drop (< 0.70) |

Command: `queue.sh jobs_A.txt` / `queue2.sh jobs_A2.txt` (2 workers), then `python evaluate.py 26 1000000`. The control `2nd_argmax` is at 0.805 here (the zoo reaches .953 at 2^30), so 2^26 understates 2^30 by roughly 0.15 on the hardest tasks.

### Stage B (zoo schedule; pending)


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
