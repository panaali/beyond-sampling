# scale_walk (T423): does the H-E16 walk quadrature keep beating sampling as H and T grow?

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · frozen recipe: origin/claude/t10-minimal-pair `m16/approaches/minimal_pair/` @ a5063c7d · compiled walk: origin/claude/t393-walk-compiled `walk_compiled/` @ c7145299 · models: origin/claude/t421-scale-study `scale_study/`

## Question
H-E16: T10's frozen deterministic walk quadrature (exact 1-D walks along fixed scrambled-Sobol tie-face lines, radial
Student-t integrand, model-free sign strata, deterministic Neyman) beats sampling at equal FPE at H16 T5, and ties at
H16 T10 (T389/T391). Does it hold, and where does it stop holding, on the T421 grid of 2nd-largest RNNs
(H in {16, 24, 32, 48, 64} x T in {10, 15, 20})? OWNER_QA 6.7.

## Pre-registration (written 2026-10-19, before any estimate is scored against a truth)
- **Estimator (frozen, no changes):** T10's `est` from `frozen/` (estimator.py, core.py, walk.py, hom.py, verbatim
  from a5063c7d; sha256 of all four asserted at import), rule seed 31415, pilot .25, called unchanged with
  `walk.walk` swapped for T393's compiled `walk_nb.walk` in `fast` mode (sha256 asserted). Nothing in the recipe is
  H- or T-specific: the face tables and strata are built per T from the code as frozen, the walk reads H from the
  weights. `max_cells` stays 4000 per line; lines that hit the cap are counted and reported.
- **Gate (before scoring):** at each grid point, on its first model, `est` with the compiled walk = `est` with the
  frozen numpy walk to <= 1e-12 relative (estimate and charged cost) at 4k FPE. Failing it, that point is scored with the
  frozen numpy walk instead.
- **Validation against brute-force MC (each model, before scoring):** (a) truth = 1e8 iid N(0, I_T) inputs, compiled
  float64 sampler (T393 `sampler.py`, copied verbatim as `sampler_t393.py`), seed crc32(tag); (b) `est` at 1M FPE
  compared with the truth: reported as z = (est - truth) / sqrt(p(1-p)/1e6). |z| > 5 or any cap hit flags the model
  (reported; the model is still scored, nothing is excluded).
- **Budgets:** 4k, 16k, 64k FPE (FPE = one forward pass of the model; the walk charges its executed steps in this
  unit, as frozen).
- **Panel per grid point:** every model registered by T421 at that (H, T) when scoring runs (both streams, cpu and
  zoo, pooled; no accuracy filter). Per-stream values are also reported.
- **Score:** F(point, budget) = mean_m[(est_m - p_m)^2 - se_m^2] / mean_m[p_m (1 - p_m) / budget], p_m the 1e8 truth,
  se_m its standard error; the estimator is deterministic so each model gives one estimate per budget. 95% CI:
  percentile bootstrap over models within the panel (10,000 resamples).
- **Holds:** strict bar if CI upper < 1; lenient bar if CI upper < 10. **Fails** a bar if CI lower > the bar; otherwise
  inconclusive at that bar.
- **Where it stops holding:** for each T, the smallest H at which a bar no longer holds; for each H, the smallest T.
  Reported per budget.
- **Scaling fit (if >= 5 grid points with both H and T varying):** OLS of log F on log H and T over the points' point
  estimates, per budget, with the bootstrap spread; otherwise descriptive only.
- **Secondary (descriptive, not scored):** the same F with 8 alternative rule seeds (31415 + 101 * 1000 * k, k = 1..8),
  as a check that the frozen rule seed is not lucky; mean cells per line, lines used per budget, cap hits.
- **Predictions [speculative]:** the prediction in OWNER_QA 6.7 is that walks degrade faster with T than with H. At
  T = 10, H16 sits near F ~ 1 (T389/T391). At H >= 24 the errors are ~10x rarer, which shrinks sampling's variance
  p(1-p)/n ~10x; the walk's error depends on how concentrated the error set is along the lines, so we predict
  F rises with H at T = 10 (lenient holds, strict fails), and the lenient bar fails at T = 20 for H >= 32.

## Method
(filled after the runs)
