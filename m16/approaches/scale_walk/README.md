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
- `frozen/`: T10's estimator.py, core.py, walk.py, hom.py verbatim from a5063c7d; `walk_nb.py`: T393's compiled walk
  verbatim; `sampler_t393.py`: T393's compiled sampler verbatim. All six sha256-asserted in `common.py`.
- `common.py`: imports the frozen `est`, swaps in the compiled walk (`fast` mode) behind a counting wrapper (lines,
  cells, cap hits; values unchanged); loads T421 models from `../scale_study` (checked out untracked by `refresh.sh`).
- `gate.py` -> `gate.jsonl`: compiled vs frozen numpy walk at 4k FPE, first model per point.
- `truths.py` -> `truths.jsonl`: 1e8-input truths. `run.py` -> `est.jsonl`: blind estimates (frozen rule + 8
  alternative rules, 4k/16k/64k; 1M validation). `analyse.py` -> `analysis.log`, `analysis.json`: the pre-registered score.
- `update.sh`: the whole pipeline for newly landed grid points.

## Status
Running: grid points (16,10), (24,10), (32,10) scored; the rest of the T421 grid is picked up as it lands.

## Results
Command: `./update.sh` (or `python analyse.py`). Full output: `analysis.log`.

**Gate:** PASS at every point checked (worst relative difference 1.2e-15). **Validation (1M FPE vs 1e8 truth):**
all 17 models |z| <= .70, zero cap hits (max cells on one line 181).

F = MSE_walk / MSE_MC at equal FPE, frozen rule seed, bootstrap 95% CI over models:

| H | T | n | 1 - acc | 4k | 16k | 64k |
|---|---|---|---|---|---|---|
| 16 | 10 | 6 | .093 | 1.18 [.48, 1.84] | 1.01 [.16, 2.21] | .33 [.19, .53] |
| 24 | 10 | 3 | .012 | .27 [.01, .52] | .29 [.12, .64] | .17 [.07, .38] |
| 32 | 10 | 8 | .008 | .41 [.08, .97] | .33 [.14, .54] | .49 [.24, .71] |

- Lenient bar (CI upper < 10): holds everywhere scored. Strict bar (CI upper < 1): holds at (24,10) and (32,10) at all
  three budgets; at (16,10) inconclusive at 4k and 16k, holds at 64k.
- **Caveat on (16,10) 64k:** the 8 alternative rule seeds give median F 1.05 [.15, 2.85], so the frozen seed's .33 is
  a favourable draw; at H16 T10 the family sits at the sampling line, as T389/T391 found. At H24/H32 the alternative
  seeds agree with the frozen one (medians .10-.29, max .99).
- Lines per 64k call fall from ~9,100 (H16) to ~3,800 (H32), and cells per line rise from ~5 to ~16, yet F improves.

## Compute accounting
- Estimator compute is charged in FPE by the frozen code (executed recurrent steps / T, plus O(H) updates and root
  tests at their FLOP share). Face tables and Sobol rules are model-free and uncharged (as frozen).
- Validation runs: 1M FPE per model. Truths: 1e8 forward passes per model (~1.5-3.5 CPU-min each at H16-32, T10).
- CPU totals so far: truths ~45 CPU-min, estimates ~3 CPU-min, gate < 1 CPU-min.

## Claims
- [measured] The frozen T10 walk quadrature, run unchanged, beats plain MC at equal FPE (strict bar) at (24,10) and
  (32,10) on the T421 panels at 4k, 16k and 64k FPE.
- [measured] At (16,10) it is at the sampling line (strict bar inconclusive; alternative rule seeds median ~1).
- [inferred] Growing H at T = 10 helps the walk relative to sampling: errors become ~10x rarer, which shrinks
  sampling's variance, but the walk's error falls faster. A plausible reason is that at higher accuracy the error set
  sits closer to the tie faces the walk integrates over, so the closed-form radial integral captures more of it.
  Not tested.
- The pre-registered prediction (F rises with H at T = 10) is **refuted** at H 16-32.

## Alternatives not ruled out
- Small panels (3 models at (24,10)); bootstrap CIs over 3 models are crude.
- Panel composition: T421's models share one training recipe; other recipes (or noisy copies, where H-E16 fails)
  could behave differently.
- The improvement with H may come from accuracy rather than H itself (H and accuracy move together here); a
  lower-accuracy H32 model would separate them.
- FPE accounting only; wall time not measured here (T393: at T10 the walk loses in wall time at 16k).

## Failure modes
- Lines hitting the 4000-cell cap would silently drop mass (none so far; reported per run).
- The method is still a fixed-node average over lines with no weight-derived bound (H-E16 scope).
