# rate_alpha — T271: convergence rates of the deterministic families, and an active-fraction-matched random control

↑ [TASKS](../../../TASKS.md) · [Fable review](../../../review/fable_review/README.md) (A4, A5, C2, D6) · [SCALING_OBSTRUCTIONS](../../SCALING_OBSTRUCTIONS.md)

**All claims unverified.** Diagnostic only.

## Pre-registration (written 2026-10-18, before any T271 number was computed)
**Question 1 (A4/C2).** For each deterministic family, how does its error shrink with compute? We fit alpha in
|bias| ~ C^-alpha (and, for families with a rule seed, RMSE over seeds ~ C^-alpha_rmse). Sampling has alpha_rmse = 0.5.
**Rule (DRI):** alpha <= 0.5 closes a family; alpha > 0.5 earns prefactor work. alpha is the OLS slope of log|bias| on log C
over >= 4 budgets spanning >= 16x. The CI is the slope ± t_{n-2, .975} x SE. A family's verdict uses the target's alpha and
the pooled median over models.

**Models:** target M_16,10 (truth = T192's 3e11-sample answer key 0.9531174789 ± 3.9e-7) and 4 leaderboard noisy copies
s0.005_0, s0.005_1, s0.01_0, s0.01_1 (truth = leaderboard 3.2e7-sample truths, SE 4-7e-5 = the bias resolution there).

**Families and their budget knob** (the knob must stay *within* one method; the v2 -> v3 -> v4 sequence in FINDINGS §5 is
three different methods, so its bias drop .74 -> .59 -> .25 pp is not a convergence rate):
- **F1 sci_tao** (fixed scrambled-Sobol face rule; `m16/approaches/sci_tao/estimator.py`, unchanged): budgets 4k, 16k,
  64k, 256k, 1M FPE (250x); 8 independent rule scrambles per budget. Bias = mean error over scrambles; RMSE over scrambles.
  Cost = charged FPE (x2.23 audit factor).
- **F2 probe v2 and v4** (`m16/estimate.py`, unchanged; weights swapped by patching its `run`/`Wo`): knob = the number of
  canonical arrangements averaged per configuration, s = 1, 2, 4, 8, 16 (cost in model evals x s). Estimate = the mean of the
  per-arrangement estimates (v4: the mean over seeds 0..s-1; v2: its own `seeds=` argument).
- **F3 FORM/SORM** (`m16/approaches/reliability/estimator.py`, unchanged): knob = `n_arr` (chamber arrangements = design
  points) 1, 2, 4, 8, 16, cost as the estimator reports.
- **F4 mass-truncated exact enumeration**, T = 4 (zoo (8,4) and (16,4) seed 0; T206 per-leaf exact data): keep the N heaviest
  leaves, count unkept mass as correct (bias = error mass of the unkept leaves). Cost = the kept leaves' mass ops + the
  mean geometry ops per leaf x N. This is an idealised best-first order (ordering by exact mass is free here). N spans >= 64x.
  (No T = 5 per-leaf data exists; noted.)
If a family is too slow for all budgets within the ~3 CPU-h cap, run >= 4 budgets spanning >= 16x on the target first and
report what was skipped.

**Question 2 (A5/D6).** Is the trained net's region count generic once the random controls match its active fraction?
Positive rescaling does not change gate patterns in a bias-free ReLU net, so activity cannot be matched by scale. Instead:
draw a pool of 400 Frobenius-matched i.i.d. random models (seeds 1000+), measure the T = 10 active fraction (1e5 inputs, the
T232 definition: the mean of h_T > 0) and keep the 5 draws closest to the trained 0.773 (within ±0.03). If fewer than 5 qualify,
use W_hh entries N(mu, 1) with mu chosen on a grid so the pool median activity is 0.77, Frobenius-rescaled. Count exact
prefix regions k = 2, 3, 4.
**Rule:** if matched-random k = 4 regions >= 1.3x trained, then training *reduces* the count relative to equally active random
nets (T232's "generic" verdict is revised). Within ±30% the T232 verdict stands. Below 0.77x, trained has more.
