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
**Done (2026-10-19).** Pre-registration cb6a7c3c; scored afterwards, no changes to recipes, runner or analysis.

## Results
**Reading: FALLING for both recipes (pre-registered rule), against the FLAT prediction.** The strict bar's DEV3 CI
upper is below 1 from 1e6 FPE on, for both recipes. On DEV3 the walk quadrature at T = 10 is compute-limited, not
method-limited, in 16k-1e7. Reported to the DRI as a possible HP5 candidate; HP5 not run.

DEV3, 100 retrains, 1e8 truths; F [95% cluster bootstrap CI]; lenient bar (F <= 10) passes everywhere.

| FPE | isg (T391) F | strict | T10 recipe F | strict | isg RMSE pp | T10 RMSE pp | truth-var share isg / T10 |
|---|---|---|---|---|---|---|---|
| 16k | 0.859 [0.659, 1.078] | no | 1.147 [0.758, 1.621] | no | .193 | .223 | .00 / .00 |
| 64k | 0.880 [0.620, 1.165] | no | 0.967 [0.651, 1.375] | no | .097 | .102 | .00 / .00 |
| 256k | 0.789 [0.534, 1.119] | no | 0.836 [0.579, 1.115] | no | .046 | .047 | .00 / .00 |
| 1e6 | **0.601 [0.434, 0.779]** | **CI < 1** | **0.590 [0.342, 0.934]** | **CI < 1** | .0203 | .0202 | .02 / .02 |
| 4e6 | **0.425 [0.297, 0.580]** | **CI < 1** | **0.469 [0.260, 0.744]** | **CI < 1** | .0086 | .0090 | .09 / .08 |
| 1e7 | **0.204 [0.080, 0.392]** | **CI < 1** | **0.357 [0.194, 0.563]** | **CI < 1** | .0038 | .0050 | .33 / .22 |

Scaling exponent beta (F ~ C^beta):

| recipe | all (16k-1e7) | per 4x | new (256k-1e7) | per 4x | reading |
|---|---|---|---|---|---|
| isg | **-0.204 [-0.307, -0.129]** | x.75 | -0.345 [-0.549, -0.212] | x.62 | FALLING |
| T10 | **-0.181 [-0.261, -0.115]** | x.78 | -0.224 [-0.378, -0.096] | x.73 | FALLING |

- 16k / 64k rows equal T391's / T10's saved DEV3 values (gate), so the low-budget end is the known ~1.
- Robustness (`robust_raw.log`): F without subtracting the truth variance (conservative: truth noise only inflates
  it): isg .611 [.444, .789] / .465 [.337, .620] / .304 [.180, .492] at 1e6 / 4e6 / 1e7; T10 .600 [.352, .944] /
  .509 [.300, .784] / .457 [.294, .663]. The strict-bar result at >= 1e6 does not depend on the truth correction.
- Bias shrinks to ~0 (isg +.031 pp at 16k → +.001 pp at 1e7): no fixed-rule offset floor is visible.
- Reproduce: `python score.py {isg,t10} B runs/{recipe}_B.jsonl` for each B, then `python analyse.py`
  (`analysis.log`, `analysis.json`); `python gate.py` (`gate.log`). Rows: `runs/`.

## Compute accounting
- Charged compute = the estimators' own executed FPE (cost column, ~= budget at every level). Per model and budget,
  process CPU with the compiled walk: .1 / .2 / .7 / 2.7 / 10.5 / 27 s (16k → 1e7). Total: 2 recipes x 100 models x
  6 budgets ≈ 2 x 41 min CPU, run as 2 processes (~45 min wall), peak RSS ~0.25 GB per process.
- Model-free work (Sobol points, face-point tables) is not charged, as in T10 / T391 (PRECOMPUTE_FLOPS = 0).
- Truths: reused T342 (1e8 samples per model); no new sampling.

## Claims
- [measured] On DEV3, F falls with budget for both frozen recipes: beta -.20 (isg) and -.18 (T10), CIs excluding 0;
  F's CI upper < 1 at 1e6, 4e6 and 1e7 FPE for both. Unverified (needs an independent check, AGENTS rule 4).
- [measured] T391's "RMSE halves for 4x" (16k → 64k) does not extend: from 256k the RMSE falls ~2.2-2.4x per 4x
  (x.75-.62 in F per 4x).
- [inferred] Item 8's METHOD-LIMITED verdict was an extrapolation from two budgets (16k, 64k) inside the noise; the
  strict bar on M_16,10 retrains is reachable by compute at ~1e6 FPE per model with these recipes. Whether ARC would
  accept a 1e6-FPE budget or the deterministic-quadrature class is a separate, open question (OWNER_QA 4.7).
- [speculative] The steeper slope at high budget (beta_new < beta_all) fits a QMC rule entering its asymptotic
  regime once the per-stratum point count is large (~1e5-1e6 points); the ~B^(-1/9) discontinuity bound in item 8
  is too pessimistic here.

## Alternatives not ruled out
- **Panel overlap:** DEV3 was the panel on which T389 / T391 / T397 were compared; the isg table is fitted on DEV2,
  not DEV3, and the T10 recipe has no DEV3-fitted part, so selection bias is small but nonzero (isg was chosen as
  "best on DEV3" at 16k/64k). The sealed HP5 check is the clean test.
- **Truth noise at 1e7:** 22-33% of the raw squared error is truth variance; subtraction is unbiased, but the 1e7
  point estimate (esp. isg .204) is noisy; the conservative F_raw still passes. 1e9 truths would sharpen 1e7.
- **One rule seed per recipe:** F at each budget is for the fixed rule; rule-to-rule spread at high budget not
  measured (T10 did 32 rule seeds at T5 only).
- **Noisy copies / leaderboard models** not scored here (the M_16,10 leaderboard groups showed larger, erratic ratios
  at 16k/64k).
- **Wall time:** at 1e7 FPE the compiled walk takes ~27 s per model vs sampling ~? (not measured here; T393 found the
  walk 2.3-3.4x slower per FPE), so a wall-time win at equal MSE needs F < ~.3-.4.

## Failure modes
- Frozen code materialises the full Sobol rule: infeasible above ~2e6 FPE under 2 GB without the lazy-rule swap.
- At 16k-256k the estimators are no better than sampling with confidence; the gain needs ~1e6 FPE per model.
- Like all walk recipes: clean models only, FPE accounting, deterministic class (H-E16 scope notes).
