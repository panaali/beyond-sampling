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

## Status
Done: negative (unverified). No readout passes.

## Results (unverified; `python m16/approaches/meanfield/readout/readout.py /tmp/claude-0/dev2/weights.npz` -> readout.json, readout_summary.json, readout.log)
50 units on 5 DEV2 models.
- "pass" = fraction of units meeting the DRI criterion (go needs >= 90%).
- The last column is the weighted mean of (P_hat - P) x 11/1024 in pp over each model's 10 units. The contribution error is
  minus this, so negative means the contribution is overestimated, as in T294/T294b. Range over the 5 models.

| method | pass | median abs err in P | branches N99 (mean / max) | (P_hat - P) x 11/1024, pp |
|---|---|---|---|---|
| (i) exact orthant (Genz), exact sampled moments | **0.36** | 0.032 | – | -0.23 … +0.03 |
| (i) Clark, exact sampled moments | 0.10 | 0.107 | – | -0.24 … +0.16 |
| (ii-c-T) m = 0 | 0.16 | 0.100 | 1 | -0.24 … +0.01 |
| (ii-c-T) m = 1 | 0.18 | 0.104 | 47 / 225 | -0.21 … +0.03 |
| (ii-c-T) m = 2 | 0.28 | 0.086 | 698 / 4,105 | -0.19 … +0.03 |
| (ii-c-T) m = 3 | **0.30** | 0.074 | 3,518 / 15,644 | -0.17 … +0.04 |
| (ii-c-G) m = 0 … 3 | 0.06 … 0.14 | 0.097 … 0.088 | m3: 7,898 / 27,833 | -0.24 … -0.08 |
| (ii-a-T / ii-a-G) m = 0 … 3 | 0.06 … 0.20 | 0.10 … 0.12 | m3: 5,384 / 31,576 | -0.26 … -0.05 |

Keeping only the top-N99 branches shifts P by <= 0.005, so it changes no verdict.

Branch counts are counted among 2e5 draws, so at m = 2–3 they are **lower bounds** limited by sample size. N99 approaches the
number of distinct patterns seen, so the tail of rare patterns is long.

## Claims
- [measured] **(i) The Gaussian assumption alone fails.** Even with the exact sampled margin mean and covariance, the exact
  Gaussian orthant meets the per-unit criterion on only 36% of units (median error 0.032 in P). Clark roughly triples that
  error (10% pass). So the margin law given a unit is far from Gaussian.
- [measured] **(ii) Exact last steps barely help.** Starting from the T294b mean-field state, treating the last m = 1–3 steps
  exactly raises the pass rate only from 16% to 30% (best variant, c with true inputs). The median error falls 0.100 -> 0.074,
  about 9% per step.
- [measured] **Branch counts grow ~5–10x per step:** N99 ≈ 50 -> 700 -> 3,500 (m = 1, 2, 3; max 15.6k).
- [inferred] **The mean-field state is already wrong at step 7–9; the error is not only in the readout.** Exact last steps on a
  wrong Gaussian state cannot recover P. The non-Gaussianity builds up over the whole recurrence. A pass would need m ≳ 5 or
  more. Extrapolating the observed branch growth (lower bounds), that means ≳ 10^5 branches per unit, i.e. far outside the
  cheap regime for 180 units. [speculative extrapolation]
- [measured] True-law inputs (T) beat Gaussian inputs (G) at every m >= 1 for variant c: the input sign leak matters too.
- [inferred] **Consequence for the T294 line:** a Gaussian-state closure of the blind region, with any readout examined here,
  does not reach per-unit fidelity. Its model-level correlation (0.64–0.74) comes from the right rough ordering, not from
  calibrated per-unit probabilities.

## Alternatives not ruled out
- **50 units on 5 models.** The units are random over the 180, not weighted by the contribution weight.
- **The oracle for (ii) is 2e5 draws from the state.** The branch sum it stands in for was not built. Its estimates carry
  ~0.001 noise, far below the observed errors.
- **The m-step truncation uses the T294b state.** A better state (a Gaussian mixture through the whole recurrence, e.g. split on
  the case-fixed step-1 gates, which are deterministic given the case) was not tried. That is the most promising untried
  variant.
- **Exhaustion checklist.** Tried: Clark, exact orthant, closures a and c, inputs G and T, m <= 3.
  - Not tried: m >= 4 (branch cost);
  - a mixture state through all steps;
  - non-Gaussian margin laws (e.g. skew-normal) on the exact sampled moments.

## Failure modes
- Long-tailed gate-pattern distributions: many rare patterns each carry little mass, but together they carry the
  misplacement.
- The Gaussian regression of h on the order-statistic inputs.

## Compute accounting
Diagnostic sampling only: 50 units x (4e5 reference + 16 x 2 x 2e5 oracle draws) forward passes of <= 10 steps. About 25 CPU-min.
No model-level estimator run, so no FPE scored.
