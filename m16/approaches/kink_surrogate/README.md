# T318: kinked-tangent surrogate (Fable r4 D2) vs the exact walk

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T312 exact walk re-implementation](../r1w_repro/README.md) · review: claude/project-handoff-priorities-zpzfm8 `review/fable_review_r4/README.md` D2

## Question
Can a first-order "kink" expansion of each line replace the exact segment walk inside the R1W V1 estimator, at a
fixed ~2.5–4 FPE per line?
- The expansion uses the first cell's value and tangent plus one adjoint per rival margin.
- Each gate-flip event (s, i) in (0, tmax) gets a location g_si = −z_si/dz_si and a margin-slope change σ·a_si·dz_si.
  Here a_si = ∂m/∂h_si from a frozen-gate adjoint, σ = +1 for off→on and −1 for on→off.

If so, the blind g12 line and extra transverse points become affordable (D1-based T317).

## Pre-registration (2026-10-18, before any run)
**Reference and estimators.** Built on my T312 re-implementation of R1W V1 (`../r1w_repro/r1w_indep.py`): closure
c, the same transverse points, v4 node, line parametrisations, tmax .5 and Gauss–Legendre masses. Only the per-line
correct set changes:
- **EXACT**: the exact segment walk (T312);
- **SURR**: pure surrogate from the first cell. Two rivals (the two smallest first-cell margins), each with its own
  adjoint. Correct iff both surrogate margins > 0, solved exactly on the piecewise-linear functions;
- **CAP1 / CAP2**: exact walk of the first 1 / 2 cells, then the surrogate restarted at the next cell's start (its
  value, tangent and adjoints);
- each in two line sets: V1's three lines (blind g12 assumed correct) and **all four lines** (blind g12 walked or
  surrogated).

**Truth for integration error:** Monte Carlo of closure c's accuracy (paired construction from x ~ N(0, I): top-3
exact, true v4 at the latest free position, the 6 other lows at μ(v4)). 4e6 samples per model, s.e. ≈ .01 pp,
seeds 318000+.

**Models:** the 10 T297 DEV2 models (indices 0, 8, …, 72).

**Reported:**
- integration RMS and mean (estimator − MC closure c) over the 10 models, per variant;
- per-line set mismatch vs EXACT: mean measure of the symmetric difference on [0, tmax];
- charged cost per line (FPE) as implemented. Surrogate: value + tangent 2 FPE, plus 1 FPE per adjoint, plus kink
  work in FLOPs. No cross-class sharing of adjoints is implemented; D2's ~2.5 FPE assumes sharing.

**Stop rule (DRI / Fable):** SURR **and** CAP2 both > .6 pp integration RMS → the first-order event expansion is
dead for the transition zone (scoped to lines, closure c, R1W V1's structure). **Go:** <= .4 pp (for SURR or
CAP2). Between: neither. The judgement uses the three-line set (V1 structure). The four-line results are reported
for T317.

**Alternatives first.**
1. Flips at tiny g change later pre-activations' slopes by O(1) relative amounts, so the dropped cross terms are not
   small (Fable "against"). CAP-c should fix most of it if the early cells dominate.
2. EXACT's own integration error vs closure c MC may already be ~.3–.6 pp (T302 measured .28 for R0). The surrogate
   must be judged against MC, not against EXACT. Both are reported.

## Status
**Done. STOP rule fired** (pre-registered): SURR 13.3 pp and CAP2 6.0 pp integration RMS, both > .6. The
first-order event expansion is **dead for the transition zone**, scoped to lines, closure c and R1W V1's structure.
Unverified.

## Results
Reproduce: `python3 run.py` (10 models, ~15 CPU-min) -> `results/run.json`, `logs/run.log`.

Implementation check: on lines whose exact walk has a single cell, SURR equals EXACT to 3e-17.

**Integration error vs 4e6-sample MC of closure c** (RMS / mean, pp; FPE per model as implemented; set mismatch =
mean symmetric-difference measure per line vs EXACT, on [0, .5]):

| variant | all 10: RMS / mean | without fused_s24: RMS / mean | mean FPE | line mismatch nb23 / nb12 / b23 (/ b12) |
|---|---|---|---|---|
| EXACT | 0.72 / +0.55 | 0.76 / +0.62 | 9,535 |  |
| SURR | 13.33 / -8.29 | 5.74 / -4.93 | 8,667 | nb23 0.036 / nb12 0.052 / b23 0.003 |
| CAP1 | 10.42 / -5.00 | 3.35 / -2.07 | 11,609 | nb23 0.020 / nb12 0.034 / b23 0.001 |
| CAP2 | 5.97 / -2.33 | 1.95 / -0.59 | 11,383 | nb23 0.011 / nb12 0.017 / b23 0.000 |
| EXACT_4L | 0.67 / +0.50 | 0.71 / +0.56 | 21,701 |  |
| SURR_4L | 13.36 / -8.35 | 5.78 / -4.99 | 11,571 | nb23 0.036 / nb12 0.052 / b23 0.003 / b12 0.083 |
| CAP1_4L | 10.45 / -5.06 | 3.37 / -2.12 | 15,963 | nb23 0.020 / nb12 0.034 / b23 0.001 / b12 0.080 |
| CAP2_4L | 5.99 / -2.38 | 1.96 / -0.65 | 16,404 | nb23 0.011 / nb12 0.017 / b23 0.000 / b12 0.073 |

**Per model** (estimate − MC closure c, pp):

| model | EXACT | SURR | CAP1 | CAP2 |
|---|---|---|---|---|
| fused_s0 | +0.74 | -5.82 | -0.94 | +0.45 |
| fused_s8 | +0.78 | -5.69 | -3.24 | -0.75 |
| fused_s16 | +0.66 | -3.10 | -0.58 | +0.52 |
| fused_s24 | -0.04 | -38.47 | -31.38 | -17.95 |
| fused_s32 | +0.46 | -4.44 | -0.07 | +0.45 |
| torch_s0 | +0.32 | -5.37 | -2.32 | -1.01 |
| torch_s8 | -0.30 | -11.43 | -8.77 | -5.61 |
| torch_s16 | +1.50 | -6.19 | -2.64 | -0.41 |
| torch_s24 | +0.61 | -1.18 | +0.23 | +0.60 |
| torch_s32 | +0.80 | -1.19 | -0.28 | +0.40 |

**Readings:**
- **The pure surrogate is wrong by −1 to −38 pp, always negative**: it predicts too many errors. CAP-1/2 help
  monotonically but stay far off on several models: CAP2 −18 pp on fused_s24, −5.6 on torch_s8, −1.0 on torch_s0.
- **The damage is on the nonblind g12 line.** Its mean set mismatch for SURR is .007–.25 (fraction of [0, .5]),
  against much smaller mismatches on the g23 lines.
- **Not a rival-choice artefact** (exploratory, `logs/rivals_explore.log`): tracking all 9 rivals instead of 2 does
  not help (fused_s24 −38.5 → −39.5; torch_s8 −11.4 → −13.2; fused_s0 −5.8 → −5.9).
- So the failure is the dropped cross terms: early flips change the *locations and slopes* of later events by O(1)
  (Fable's "against" argument, confirmed).
- **Cost is also no saving as implemented:**
  - SURR costs 8.7k FPE per model vs EXACT's 9.5k: 4 FPE per line without sharing, against the walk's ~4.4.
  - CAP2 costs more than EXACT.
  - D2's ~2.5 FPE per line needs cross-class adjoint sharing, which would not fix the accuracy.
- **Side finding:** the EXACT walk's integration error against closure c's MC is .72 pp RMS (+.55 mean) on these 10
  models. That is larger than T302's .28 for R0. V1's declared approximations (v4 node, v4 moving with v3, product
  set, one transverse point) cost about .7 pp before any surrogate. It matches R1W's README inference of an
  integration component of ~.6 pp.

## Claims
- K1 [measured] The kinked-tangent surrogate (first cell + 2 adjoints) and its CAP-1/2 hybrids misestimate closure c's
  accuracy by 13.3 / 10.4 / 6.0 pp RMS (10 DEV2 models), always too low. The pre-registered stop rule fires.
- K2 [measured] The failure is not the rival set (all 9 rivals give the same) but the first-order treatment of flip
  interactions, concentrated on the nonblind g12 line.
- K3 [measured] The exact walk itself is .72 pp RMS from closure c's MC (+.55 mean): V1's integration error is ~.7 pp,
  not ~.3.

## Compute accounting
10 models × 8 estimator variants (~0.2 s each) + 4e6-sample closure MC per model (~40 s each), about 10 CPU-min;
the exploratory rival check took ~1 min.
