# walk_budget (T429): does walk-quadrature F fall with budget at T = 10?

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [COMPUTE_FRONTIER item 8](../../../docs/COMPUTE_FRONTIER.md) ·
frozen recipes: origin/claude/t10-minimal-pair `m16/approaches/minimal_pair/` @ e3a83b95 · compiled walk: T393 @ c7145299

## Question
At ARC's T = 10 (M_16,10 retrains), does the frozen deterministic walk-quadrature estimator's F = MSE x C / p(1-p)
fall as the budget C grows? Flat F means the strict bar (F <= 1) is method-limited; falling F means it is
compute-limited. On DEV3, F sits at ~1 at 16k-64k FPE for every recipe tried (T10 recipe, T389, T391, T397).

## Pre-registration (written 2026-10-19, before any scoring at a new budget)
- **Recipes (frozen, sha256[:12] asserted in `score.py`):**
  - **primary: T391 `isg`** (`frozen/isg/estimator.py` 0bb493e6b480 + `frozen/pos/table.npz` e772b13f333a, RULE_SEED
    14142): the best DEV3 point estimates at both 16k and 64k so far;
  - **secondary: the T10 recipe** (`frozen/estimator.py` fb7902e12240, RULE_SEED 31415), the one item 8 says falls
    x.84 per 4x.
  - Shared: `walk.py` a38258f586b4, `hom.py` a91f019fca21, `core.py` 48dbba49ab13, all verbatim from e3a83b95.
- **Execution-only swaps** (`score.py`), both passed by `gate.py` before any new-budget run (`gate.log`):
  1. `core.sobol` returns a lazy rule generating the requested slice of the same scrambled Sobol sequence (the frozen
     code materialises 2^ceil(log2(B/2+64)) x 10 points per stratum: ~3.4 GB at 1e7 FPE, over the 2 GB rule).
     Bitwise equal to the frozen rule; with the numpy walk the estimates equal the saved T391 / T10 DEV3 rows bitwise.
  2. T393's compiled walk (`frozen/walk_nb.py` bbc6cc992c7c, repro mode: scipy Student-t CDF as frozen). Worst
     relative difference to the saved DEV3 rows, 100 models x {16k, 64k} x both recipes: 1.7e-15.
- **Panel:** DEV3 (T342: 100 open M_16,10 retrains, weights sha256 62c95203..., truths abcb7d6d..., 1e8 samples each).
  HP5 (sealed reserve) is not touched.
- **Budgets:** 256k, 1e6, 1e7 FPE (the task), plus 4e6 (cheap, for the fit); 16k and 64k re-run with the same runner
  (equal to the saved rows by the gate). One deterministic run per model per budget.
- **F**, exactly as T389 / T391 (`score_dev3.py`): F = mean_m[(est - p)^2 - se_truth^2] / mean_m[p(1-p)/cost], cost as
  executed; 95% CI by cluster bootstrap over models (2000 resamples, rng 0). Reported against both bars: strict F <= 1
  and lenient F <= 10 ([PROTOCOLS Directive 8](../../../docs/PROTOCOLS.md)); a bar counts as cleared only if the CI
  upper bound is below it. Truth noise: at 1e7 FPE the 1e8 truths' variance is ~10% of sampling's p(1-p)/C; it is
  subtracted (unbiased), and its share of the raw squared error is reported per budget.
- **Scaling exponent:** beta = slope of log F on log C (least squares over the budgets), with a bootstrap CI (the same
  model resamples at every budget). Two fits: "all" (16k-1e7) and "new" (256k-1e7). Primary: "all", isg.
- **Reading rule (per recipe, primary fit):**
  - **FALLING** (compute-limited) iff the beta CI upper < 0;
  - **RISING** iff the beta CI lower > 0;
  - **FLAT** otherwise (CI contains 0): the strict bar is method-limited in this range.
  - If any budget gives a DEV3 CI upper < 1 (either recipe), that budget is reported to the DRI as a possible HP5
    candidate; HP5 is not run here.
- **Predictions:** isg FLAT (beta ~ 0, F ~ .8-.95 at every budget; T391's RMSE halved for 4x); T10 recipe beta ~ -.13
  (x.84 per 4x, item 8), so F ~ .6-.7 at 1e7, CI likely still touching 1 → no strict pass expected at 1e7.
  Item 8's ‡ argument (~B^(-1/9) for a discontinuous ~9-dimensional integrand) predicts beta ~ -.11 for any fixed QMC
  rule if the error is in the discontinuity, and beta ~ 0 if it is dominated by an un-randomised fixed-rule offset.
- **Threats noted in advance:** (1) a deterministic rule has one error per model; F mixes a "bias" (fixed-rule offset)
  and per-model scatter: bias in pp is reported per budget; (2) Neyman allocation is re-run per budget (its pilot
  is 25% of the budget), so budgets are not nested prefixes; (3) the bootstrap CI ignores truth-error uncertainty.

## Method
- `score.py RECIPE BUDGET OUT.jsonl`: one `est` call per DEV3 model, rows to `runs/RECIPE_BUDGET.jsonl` (resumable).
- `gate.py`: reproduction gate (above). `analyse.py`: the table and fits (`analysis.json`, `analysis.log`).

## Status
Pre-registered; scoring not started.
