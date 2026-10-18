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
Pre-registered.
