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

## Pre-registration (written 2026-10-19, before any fresh model was trained or any score computed)

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
To be filled after the runs. Files: `train_fresh.py` (training), `frozen/` (T10's estimator, verbatim).

## Status
Pre-registered; training running.

## Results
None yet.

## Compute accounting
To be filled.

## Claims
None yet.

## Alternatives not ruled out
To be filled.

## Failure modes
To be filled.
