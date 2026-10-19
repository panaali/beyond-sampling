# minimal_pair_check (T379): does T10's frozen H16 T5 estimator beat sampling on FRESH models?

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md) ·
[minimal_pair (T10, branch claude/t10-minimal-pair)](../minimal_pair/README.md)

## Question
T10 reports that its frozen estimator (`minimal_pair/estimator.py`, commit a5063c7d) beats sampling at H=16, T=5 on
zoo seeds 0-4: per-seed MSE ratio <= 0.23 at 16k and 64k FPE. Its sign strata were chosen after diagnostics on those
same five seeds. Does the result hold on fresh H=16, T=5 models that T10 never saw? Secondary: is its 30-37x CPU-time
ratio against sampling only Python per-cell overhead?

## Class
The DRI ruled the estimator **deterministic QMC-class quadrature over exact 1-D walks**: fixed scrambled-Sobol nodes
(model-independent), each node integrated exactly along a 1-D line by the ReLU-region walk. It uses no random model
inputs, but it is not mechanistic. This check does not change that ruling.

## Verdict
**GO (pre-registered line met).** On 5 fresh H=16, T=5 models that T10 never saw, T10's frozen estimator (unchanged,
sha256-asserted) has per-model MSE ratio 0.00-0.12 at 16k and -0.00-0.14 at 64k, all far below 1. Over 32
alternative fixed rules its gain over sampling is 6.7-34x (16k) and 9.6-28x (64k), at least as large as T10's 5.9-14x
on the zoo seeds. So choosing the sign strata on the test seeds did not inflate the T=5 result. The noisy copies are a
separate story: the recipe loses to sampling at sigma = 0.02 (group ratio 2.8 at 16k, 1.7 at 64k), as T10 reported
for the zoo copies.

## Pre-registration (written 2026-10-19, before any fresh model was trained or any score computed; commit 194f543c)

### Models
- **Fresh clean models.** No unused H=16, T=5 zoo models exist (the zoo has only seeds 0-4 at full training, and
  its partial checkpoints are the same training runs). So I retrain with `alg_zoo/training.py` (`train_fresh.py`),
  zoo recipe, all defaults (n_train 2^30, batch 1024, AdamW lr 0.04/sqrt(16), weight decay 1e-3, no bias), on CPU.
  - Primary seeds: **100, 101, 102, 103, 104**.
  - Reserve seeds, used in this order only to replace a primary model that is not zoo-like: 105, 106, 107.
- **Zoo-like.** Error rate (own truth) in [0.5%, 2.0%]. The zoo seeds 0-4 have 0.75-1.28% (leaderboard truths).
  A model outside the range is still reported, with its scores, in a separate row.
- **Noisy copies.** For each fresh clean model, the arc_eval / leaderboard recipe: W + sigma * RMS(W) * N(0,1) on each
  matrix, sigma in {0.0025, 0.005, 0.01, 0.02}, 2 copies per sigma, numpy `default_rng(1000 + model seed)`.
  40 noisy models, reported separately; **not in the go line**.

### Estimator
- T10's files copied verbatim from commit a5063c7d into `frozen/` (estimator.py, core.py, walk.py, hom.py). Every
  scoring script asserts their sha256 before importing:
  - estimator.py `fb7902e12240acd86dc6cd18786018abe73201a1996357efead379bd487fa867`
  - core.py `48dbba49ab13367615e63a396d56f506bba9f8adda76ced4886a6023ecef8283`
  - walk.py `8cdf611a37348f87b4c6947710ffd52f3f89286d52262884870d5f763066344b`
  - hom.py `a91f019fca215b5b95996f77ff514811679ffda645112f010a0ec780a78b446d`
- Call: `est(model, budget, rng)` with default arguments (RULE_SEED 31415), budgets **16000 and 64000**. One run per
  (model, budget), because the estimator is deterministic. Determinism is checked with 2 different rngs per cell.
- The shared harness cannot load these models (`leaderboard.models` and `arc_eval.noisy_models` are tied to the zoo
  and M_16,10), so I score with the leaderboard's formula in my own script.

### Truths (own code)
- Own numpy float64 forward pass (`truth.py`), checked against the torch module on 1e6 inputs.
- **1e8 iid N(0,1) samples per clean model**, 1e7 per noisy copy, fixed seeds; se = sqrt(p(1-p)/n).

### Metric and go line
- Per-model MSE ratio r = ((est - p)^2 - se^2) / (p (1 - p) / cost), cost = the estimator's charged cost (the
  leaderboard formula, as in T10's `lb_perseed.py`).
- **GO: r < 1 at both 16k and 64k on every one of the 5 fresh clean (zoo-like) models.** Anything else is a FAIL for
  "the T10 recipe beats sampling on fresh H16 T5 models", and I report which models and budgets fail.
- Reported but not in the go line:
  - noisy copies, per sigma group;
  - a robustness study over 32 alternative fixed rules (`rule_seed = 7000 + 13 k`, k = 0..31, as in T10's
    `mse_study.py`), gain = sampling MSE / rule MSE with a bootstrap 95% CI over rules;
  - the bias z-score of the 32-rule mean.

### CPU-time audit (not scored)
- Measure CPU seconds of `est` and of plain vectorised sampling at the same cost (single thread), at 16k and 64k.
- Profile `est` (cProfile): the share of time in numpy kernels vs Python-level code.
- Estimate a vectorised walk's cost: time the arithmetic the walk is charged for (the same matmul/O(H) operation
  count, in large batches) and compare to sampling at equal FPE. The scored estimator is not changed.

## Method
- `train_fresh.py SEED`: trains one model with `alg_zoo.training.train("2nd_argmax", hidden_size=16, seq_len=5,
  seed=SEED)`, all defaults (the zoo recipe), single CPU thread. Checkpoints in `data/t379_models/` (git-ignored).
- `models.py`: loads the final checkpoints as float64 (Wi, Wh, Wo); makes the noisy copies (pre-registered recipe).
- `truth.py`: own float64 forward pass, iid N(0,1) inputs, 1e8 per clean model and 1e7 per noisy copy. The rng seed
  is crc32 of the model name. On 1e6 inputs per clean model it agrees with the torch module's argmax on 100%.
- `frozen/`: T10's estimator.py, core.py, walk.py, hom.py, verbatim from a5063c7d. `frozen_check.py` asserts their
  sha256 before import.
- `score.py`: the frozen rule on every model at 16k and 64k, leaderboard MSE-ratio formula, determinism check (2 rngs),
  CPU time vs own vectorised sampling. `score.py rules 32`: the 32-rule robustness study. `summarize.py`: group table.
- `cpu_audit.py`: the CPU-time audit (profile, walk counters, large-batch run of the same walk code).
- Process note: a container restart killed the first training runs of seeds 102 and 103; they were retrained from
  scratch with the same seeds. Seeds 100-104 were all zoo-like, so no reserve seed was used.

## Status
Done (2026-10-19). Verdict GO for the clean fresh models; noisy copies reported separately.

## Results

### Fresh models (own truths)
Validation accuracy from training (2^24 samples) agrees with the own truth to 1e-4 on every model.

| model | own truth acc (1e8) | error | zoo-like? |
|---|---|---|---|
| f100 | 0.990218 (se 1.0e-5) | 0.98% | yes |
| f101 | 0.989769 (se 1.0e-5) | 1.02% | yes |
| f102 | 0.993128 (se 0.8e-5) | 0.69% | yes |
| f103 | 0.992816 (se 0.8e-5) | 0.72% | yes |
| f104 | 0.991686 (se 0.9e-5) | 0.83% | yes |

The zoo seeds 0-4 have 0.75-1.28% (leaderboard truths), so the fresh models are on average slightly more accurate.
Reproduce: `python train_fresh.py S` (from repo root, S = 100..104), then in this folder `python truth.py`.

### Pre-registered go line: frozen rule, one deterministic run per model
Reproduce: `python score.py` (-> `score.json`, `score.log`).

| model | 16k: est | cost | **MSE ratio** | 64k: est | cost | **MSE ratio** |
|---|---|---|---|---|---|---|
| f100 | 0.990149 | 16975 | **0.008** | 0.990073 | 64031 | **0.139** |
| f101 | 0.989758 | 16209 | **0.000** | 0.989687 | 63624 | **0.041** |
| f102 | 0.993255 | 16009 | **0.037** | 0.993127 | 63700 | **-0.001** |
| f103 | 0.993038 | 16126 | **0.111** | 0.992855 | 63823 | **0.013** |
| f104 | 0.991929 | 17134 | **0.122** | 0.991647 | 63686 | **0.011** |

**All 10 < 1: GO.** Determinism check: every cell gave identical (estimate, cost) under 2 different rngs (`det_ok`).
For comparison, T10's leaderboard per-seed ratios on zoo seeds 0-4 were 0.01-0.23 (16k) and 0.00-0.22 (64k).
Sanity check: on zoo seed 0, my copy of the frozen estimator reproduces T10's leaderboard estimate exactly
(0.9902911726804176 at 16k, cost 16113.02).

### Robustness: 32 alternative fixed rules (rule_seed 7000 + 13k), clean fresh models
Reproduce: `python score.py rules 32` (-> `rules.json`, `rules.log`). Gain = sampling MSE / rule MSE, bootstrap 95%
CI over rules; bias z = (mean of 32 estimates - truth) / its s.e.

| model | 16k gain [CI] | bias z | 64k gain [CI] | bias z |
|---|---|---|---|---|
| f100 | 8.2 [5.9, 13] | -0.1 | 9.6 [6.4, 16] | +0.1 |
| f101 | 6.7 [4.5, 11] | +0.3 | 23.6 [15, 42] | -0.3 |
| f102 | 34.2 [25, 53] | +0.9 | 28.2 [20, 44] | **+2.1** |
| f103 | 14.6 [10, 24] | +1.2 | 27.2 [20, 42] | +1.6 |
| f104 | 9.6 [6.2, 18] | -0.2 | 10.1 [7.0, 17] | -0.7 |

T10 on zoo seeds 0-4: 5.9-10.4x (16k), 8.7-14.1x (64k). The fresh models give equal or larger gains, in line with
their lower error rates (T10's law: gain ~ 1/err at roughly constant CV). The 32-rule means show a small positive
bias on f102/f103 at 64k (z 1.6-2.1; +2.2e-5 to +2.8e-5 in accuracy), about 8% of the sampling s.e. at 64k (3.3e-4), so bias^2 is <1% of
sampling's MSE.
- Per-run ratios over all 320 rule runs (5 models x 2 budgets x 32 rules) are all < 1 (max 0.97, f101 at 16k). The frozen
  rule sits at the 0-84th percentile of its model's 32-rule spread, i.e. it is a typical rule, not a lucky one.

### Noisy copies (not in the go line)
Reproduce: `python summarize.py` (-> `summary.log`). Group ratio = mean(se2)/mean(p(1-p)/cost) over 10 copies (5
models x 2), bootstrap CI over copies; one deterministic run per copy.

| group | accuracy range | 16k ratio [CI] | copies >= 1 | 64k ratio [CI] | copies >= 1 |
|---|---|---|---|---|---|
| clean | 0.990-0.993 | 0.05 [0.01, 0.10] | 0/5 | 0.05 [0.01, 0.10] | 0/5 |
| s0.0025 | 0.933-0.989 | 0.46 [0.10, 0.74] | 1/10 | 0.14 [0.02, 0.23] | 0/10 |
| s0.005 | 0.854-0.983 | 0.60 [0.03, 1.00] | 2/10 | 0.14 [0.07, 0.20] | 0/10 |
| s0.01 | 0.697-0.944 | 0.86 [0.30, 1.76] | 2/10 | 0.22 [0.04, 0.41] | 0/10 |
| s0.02 | 0.497-0.887 | **2.76 [0.39, 5.50]** | 3/10 | **1.72 [0.59, 3.08]** | 5/10 |

- The estimator's advantage shrinks as the error rate grows and is lost at sigma = 0.02 (T10: 1.46 at 16k, 0.73 at
  64k on the zoo copies). This matches T10's own failure mode "bulk errors: gain ~ 1/err".
- The fresh models are more noise-sensitive than the zoo base: f100's copies fall to 0.93 / 0.85-0.90 / 0.70-0.73 /
  0.50-0.59 accuracy, while the leaderboard's zoo copies stay at 0.977-0.982 / 0.956-0.966 / 0.91-0.93 / 0.82-0.86.
  So the noisy groups here are harder than the zoo's at the same sigma and are not directly comparable.

### CPU-time audit (`python cpu_audit.py 100` -> `cpu_audit_100.json`; not scored)
Time ratio = CPU s of est / (charged cost x CPU s per FPE of own vectorised sampling at that batch size), one thread.

| budget | time ratio | walk() calls / loop iterations / row-cells | share of time in walk() |
|---|---|---|---|
| 16k | 27 (19-35 over the 5 models) | 20 / 683 / 11.7k | 79% |
| 64k | 10 (10-16) | 30 / 1158 / 44k | 93% |

- **Large-batch floor.** The same walk code run once on 65,536 faces per stratum (one call, few iterations per row)
  costs 6.1e-6 CPU s per charged FPE, vs 7.1e-7 for sampling: **ratio 8.6.** That is what numpy vectorisation of
  this walk achieves with per-call overhead removed.
- **Decomposition at 16k (f100, 0.37 s).**
  - ~0.10 s: marginal walk work (11.7k row-cells at the large-batch rate);
  - ~0.19 s: per-iteration fixed cost (683 loop iterations on small active sets, ~0.28 ms each);
  - ~0.08 s: outside the walk (face-point maps, Sobol, Student-t CDF).
- **So the 30-37x is only partly Python overhead** [measured]:
  - per-iteration overhead accounts for ~3x at 16k and ~1.2x at 64k;
  - the remaining ~8.6x persists at any batch size.
- **What is the ~8.6x?** [inferred from reading walk.py]
  - It is not much uncharged arithmetic. Per row-cell the code does about 21·T·H element operations of O(H) work
    (gathers/scatters of the stored Z, DZ; linear update; gate test; root test; ReLU) against the 14·T·H it
    charges. That is ~7·T·H = 560 uncharged element ops against ~3.8k charged FLOPs per row-cell, i.e. about
    1.15-1.3x.
  - The rest is throughput. Sampling spends its FLOPs in (n x 16) @ (16 x 16) BLAS calls; the walk spends them in
    masked fancy-index gathers, boolean selects and small elementwise numpy passes over (m, T, H) arrays.
- **Vectorised-walk estimate** [inferred]:
  - a numpy implementation with all faces batched (no pilot chunking) would sit near the measured 8.6x;
  - a compiled walk (numba/C, one pass per row) should reach roughly the charged-FLOP rate, ~1.5-3x sampling's time
    per FPE.
  - Not built: the task forbids changing the scored estimator, and a separate prototype would not be the scored
    entry.
- Conclusion for leaderboard rule 6: the high time ratio does not indicate a large under-count of arithmetic
  (at most ~1.3x by the code count above). It is numpy small-batch overhead plus irregular-access throughput.

## Compute accounting
- Unit: one forward pass of one length-5 sequence (FPE). The estimator's charged cost is T10's (walk.py), unchanged.
- Training: 5 models x 2^30 sequences (+2^24 validation), plus ~1.4 lost partial runs (container restart);
  ~2.1 ms CPU per 1024-sequence step, ~37 CPU min per model.
- Truths: 5 x 1e8 + 40 x 1e7 = 9e8 sequences (own forward), plus 5 x 1e6 for the torch agreement check.
- Scoring: 45 models x 2 budgets x 2 runs (determinism) of ~16k / 64k FPE; 32-rule study 5 x 2 x 32 runs: ~1.4e7 FPE.
- CPU audit: 6 estimator runs + 5 x 65,536 faces (~4.6e6 FPE) + sampling timing.

## Claims
- [measured] T10's frozen estimator, unchanged, beats sampling on 5 fresh H16 T5 models at 16k and 64k: per-model MSE
  ratio <= 0.14, against own 1e8-sample truths. The pre-registered go line is met.
- [measured] Over 32 alternative fixed rules the gain on the fresh models is 6.7-34x (16k) and 9.6-28x (64k).
- [inferred] Choosing the sign strata on zoo seeds 0-4 did not inflate T10's T=5 result: the held-out gains are at
  least as large. The fresh models are slightly more accurate than the zoo seeds, which by T10's law favours the method.
- [measured] On noisy copies the advantage shrinks with the error rate and is lost at sigma = 0.02 (group ratio 2.8 /
  1.7). The fresh models' noisy copies are less accurate than the zoo's at equal sigma.
- [measured] The CPU-time ratio is 19-35x at 16k and 10-16x at 64k on the fresh models. Per-iteration overhead
  explains ~3x at 16k; a single-batch run of the same numpy walk still costs 8.6x sampling per FPE.
- [inferred] A compiled walk would bring the time ratio to ~1.5-3x; uncharged arithmetic is at most ~1.3x.

## Alternatives not ruled out
1. **Training differs from the zoo's.** These models were trained on CPU with this torch version; the zoo was
   probably trained on GPU. Same recipe, different numerics. The fresh models are slightly more accurate (0.69-1.02% vs
   0.75-1.28% error) and their noisy copies lose more accuracy. A lower error rate favours the method, so the fresh
   set may be slightly easier than the zoo. A harder held-out set (e.g. partially trained checkpoints, or
   T10's T=6 rung, T11) would test the margin.
2. **One deterministic run per model.** The per-model ratio is a single draw of the quadrature error; the 32-rule
   study shows the frozen rule is typical (0-84th percentile of each 32-rule spread; all 320 rule runs < 1).
3. **Small positive bias at 64k on f102/f103** (32-rule mean z 1.6-2.1, ~+2.5e-5). It is ~8% of sampling's s.e. and does not
   affect the verdict, but it could matter at much larger budgets.
4. **Same architecture family.** Fresh seeds test the choice of strata, not the method's assumptions (no biases,
   sign symmetry). A model with biases would break the radial integration (T10's failure modes).
5. **The CPU audit's compiled-walk figure is an estimate**, not a measurement. A numba prototype would settle it.

## Failure modes
- Bulk errors (noisy copies at sigma = 0.02 here; T10's law gain ~ 1/err): the method loses to sampling.
- High CPU time at small budgets in this numpy implementation (pilot chunks of 16 faces -> many small iterations).
- Assumes a bias-free ReLU RNN (radial integration, sign strata).
