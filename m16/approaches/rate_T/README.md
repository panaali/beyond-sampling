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
