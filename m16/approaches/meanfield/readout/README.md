# readout — T295: is the blind-region error the readout? exact orthant, and mean-field + exact last m steps

↑ [meanfield (T294/T294b)](../README.md) · [TASKS](../../../../TASKS.md)

**All claims unverified. Diagnostic only (sampling is used only as the reference and as an oracle for exact integrals).**

## Question
T294b found that a Gaussian-logit + Clark readout misplaces the per-unit P(correct) even when it is fed the exact sampled logit
moments. This task asks which readout reaches the sampled per-unit P:
- (i) the exact Gaussian orthant;
- (ii) a mean-field Gaussian state through step 10-m, then exact treatment of the last m steps by gate-pattern branches, then an
  exact orthant.

## Pre-registration (written 2026-10-18, before any T295 number was computed)
**Units.** 50 blind units: DEV2 models #0, #5, #20, #40, #60 (weights from t192 hp2/dev2), each with 10 units.
- Units 1–5 of models #0 and #5 are the T294b diag units.
- All other units are drawn uniformly from T293's 180 (case, roles, label) units with rng 295.
- Reference P(correct | unit): 4e5 exact samples of the unit's input law (as in diag.py).
**Methods.**
- (i) **Exact orthant on the exact sampled margin moments.** P(N(m, C) >= 0) by Genz (scipy multivariate_normal.cdf,
  abseps 1e-5). Also reported: Clark on the same moments (isolates Clark from the Gaussian assumption).
- (ii) **MF + exact last m steps, m = 0, 1, 2, 3.**
  - The T294b closure (variant c: exact bivariate cross moments plus a sign-fixed binned x_t; variant a reported as secondary)
    runs through step 10-m and gives a Gaussian state over (x, h_{10-m}).
  - The last m steps are the exact network. The value is the integral, over that state, of the indicator "argmax = label".
  - Its deductive form is a sum over the gate patterns of the last m steps (16m sign bits). Each branch is a Gaussian polyhedron
    with Gaussian mass and an exact orthant.
  - In this diagnostic the integral is evaluated by 2e5 draws from the Gaussian state (an oracle for the branch sum, not the
    estimator).
  - Two input treatments for the last m steps:
    - **(ii-G)** the inputs x_{10-m..9} come from the Gaussian state itself;
    - **(ii-T)** they are drawn from their true sign-fixed law (the unit's sampler), with h_{10-m} drawn from its Gaussian
      regression on x.
  - m = 0 is the exact orthant on the closure's own moments.
- **Branches.** For each m, count the distinct last-m gate patterns among the oracle draws.
  - N99 and N999: the number of patterns covering 99% and 99.9% of the mass.
  - Report the error when only the top-N99 patterns are kept (renormalised).
  - Report the mean over units, and the maximum.
**Per-unit criterion (DRI):** |P_hat - P| <= 0.2 P, or |P_hat - P| <= 0.01 where P < 0.05.
**Reported per method:**
- the fraction of the 50 units that pass;
- the median |P_hat - P|;
- the weighted blind-contribution error per model (pp; weights as in T293, over the 10 units, diagnostic).

A method **passes** if >= 90% of units meet the criterion. The deliverable is error vs m plus branch counts. No model-level run.
Cap ~2 CPU-h.
