# rate_T — T277: does sci_tao's convergence rate fall toward 0.5 as T grows (dimension factor for the H-E11 collapse)?

↑ [TASKS](../../../TASKS.md) · [rate_alpha (T271)](../rate_alpha/README.md) · [Fable review](../../../review/fable_review/README.md) A4

**All claims unverified.** Diagnostic only.

## Pre-registration (written 2026-10-18, before any T277 number was computed)
**Hypothesis (Fable A4).** Deterministic face quadrature has error ~ N^(-s/d). Its integrand F lives on a (T-1)-dimensional
face and has kinks (s ~ 1). So sci_tao's rate alpha_rmse(T) should start above 0.5 at small T and fall toward the RQMC/sampling
floor 0.5 as T grows. That would explain H-E11's collapse (507x at T = 3 -> 1.6x at T = 10) without "heavy tails" as a
primitive. T272 refuted "kink density grows with T"; this test leaves kink density aside and measures the rate itself.

**Measurement.** sci_tao estimator (`m16/approaches/sci_tao/estimator.py`, unchanged) on `alg_zoo.zoo_2nd_argmax(16, T, seed)`,
seeds 0, 1, 2, T = 3..10. Budgets 4k, 16k, 64k, 256k, 1M FPE (250x); 8 independent rule scrambles per budget (rule seeds
4000 + 10 s, as T271). Truth: 4e7-sample MC per model (rng 9000 + 100 T + seed, chunks of 5e5; SE ~1-3e-5). MSE at a budget =
mean over scrambles of (err^2 - truth SE^2). alpha_rmse = OLS slope of log sqrt(MSE) on log charged cost (x2.23 audit
factor), with a 95% CI from the slope SE. Per T: the median over seeds and a pooled fit (log-mean MSE over seeds). Points with MSE
<= 0 after the truth correction are dropped and counted.
Also mass-truncated exact enumeration (T271 F4) at T = 3, 4 (zoo seed 0, saved T206 per-leaf data). No T = 5 per-leaf final data
exists (T192's k5 masses are prefix regions without labels), so T = 5 is not possible.

**Decision rule.**
- **Supported** if the pooled alpha_rmse(3) >= 0.75, alpha decreases with T (Spearman rho(T, alpha) <= -0.6 over the 8 T
  values), alpha(10) is within 0.1 of 0.5, and a fit alpha(T) = 0.5 + s/(T - 1) has an s whose CI excludes 0.
- **Refuted** if alpha(T) is flat (all pooled values within 0.5 ± 0.15) or does not decrease (rho > -0.2).
- Otherwise **mixed**, reported with which part failed.
**Alternatives.** Scramble count (8) limits each MSE to roughly ±50%. Truth SE limits the large-budget points at small T, where the
error is smallest. Neyman allocation changes with budget (pilot effect), so the "rate" also includes allocation effects.

**Amendment (2026-10-18, before any sci_tao row was computed):** at T = 3, 4 the zoo models' error is so small (accuracy
0.9994 / 0.998) that sci_tao's large-budget error falls below the 4e7-sample MC SE. For T = 3, 4 the truth is therefore T194's exact
enumeration accuracy (width16_probe/res/H16_T{3,4}_zoo{s}_exact.json; leaf sums within 5e-10, validated against MC), with SE 0.
MC truths are still computed and stored for all T.

**Amendment 2 (2026-10-18, data-independent):** alg_zoo has no zoo model at T = 7 or 9 (it offers 2, 3, 4, 5, 6, 8, 10), so the
T set is {3, 4, 5, 6, 8, 10}. The Spearman rule (rho <= -0.6) is applied over these 6 values.

## Results (unverified; `run.py`, `fit.py` -> fit.json / fit.out, `prefactor.json`, `enum.json`)
720 sci_tao runs: 6 T x 3 seeds x 5 budgets x 8 scrambles. Truths: exact at T = 3, 4; 4e7-sample MC at T = 5-10.

| T | pooled alpha_rmse [95% CI] | per-seed alpha | median MSE ratio @16k | @256k |
|---|---|---|---|---|
| 3 | 0.64 [0.50, 0.77] | [0.66, 0.55, 0.67] | 0.000981 | 0.000993 |
| 4 | 0.64 [0.57, 0.70] | [0.66, 0.76, 0.57] | 0.03 | 0.019 |
| 5 | 0.49 [0.39, 0.58] | [0.39, 0.87, 0.63] | 0.254 | 0.221 |
| 6 | 0.61 [0.50, 0.73] | [0.64, 0.6, 0.65] | 1.69 | 0.323 |
| 8 | 0.56 [0.49, 0.63] | [0.48, 0.55, 0.62] | 1.08 | 0.654 |
| 10 | 0.67 [0.54, 0.79] | [0.68, 0.72, 0.48] | 2.49 | 0.782 |

Spearman rho(T, alpha) = **0.03**; fit alpha = 0.5 + s/(T-1): s = 0.32 [0.05, 0.60]. That fit forces the
intercept to 0.5; the data are flat at ~0.6 instead.
Mass-truncated exact enumeration (zoo seed 0, saved T206 leaves): alpha_bias 1.07 [0.59, 1.55] at T = 3 and 1.40 [0.72, 2.07] at T = 4.
These are finite methods that converge to exactness, so they do not test the hypothesis. MSE ratio at T = 3 is <= 0.01 at every
truncation from 1/64 of the leaves.

## Verdict (pre-registered rule): **refuted**
alpha(3) = 0.64 < 0.75 and rho = 0.03 > -0.2. sci_tao's convergence rate does **not** depend on T: it is ~0.5-0.67 (the RQMC regime,
slightly better than sampling's 0.5) at every T from 3 to 10. The H-E11 collapse is a **prefactor** collapse. At equal budget
sci_tao's MSE ratio to sampling rises ~1000x from T = 3 (~1e-3) to T = 6-10 (~0.3-2.5), at an unchanged rate. The dimension
factor N^(-s/d) predicts a falling exponent and is not what happens. What grows with T is the integrand's variance relative to
p(1-p): more and wider error bands per face, which matches the heavy tails of H-E12 and T218 [inferred].
Side observation: on these zoo seed models sci_tao's ratio at 256k is < 1 at every T (0.78 at T = 10). On the target it is ~1
(T271). So the clean zoo seeds are slightly easier than the target at T = 10 [measured, 3 seeds].

## Alternatives not ruled out
- 8 scrambles per point: per-point MSE is uncertain to ~±50%, and the per-seed alphas scatter (0.39-0.87 at T = 5).
- The pilot-based Neyman allocation changes with budget, so alpha mixes rule convergence and allocation; a fixed-allocation run
  would isolate the rule.
- Budgets 4k-1M only. An asymptotic rate change beyond 1M (e.g. once all kinks are resolved) is not excluded.
- Truth MC SE (1.6-5e-5 at T >= 5) is subtracted; points with MSE <= 0 after correction were dropped (none at T >= 5 with
  this budget range, per fit.json).

## Compute accounting
Truths 4e7 x 18 models (T >= 5; ~1 CPU-h with T = 3, 4 MC kept for reference); sci_tao 720 runs (~50 CPU-min). ~2 CPU-h total,
single thread.
