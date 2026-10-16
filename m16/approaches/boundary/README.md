# Boundary-layer (tie-surface) integration estimator for M_16,10 accuracy

## Question
Can the error volume of M_16,10 be rewritten as integrals over the tie surfaces of the top order statistics
(x_(2)=x_(3), x_(1)=x_(2)), with the normal direction integrated **exactly** through the network's
piecewise-linear structure, so that an estimate beats plain random sampling at equal compute under ARC's
protocol (`m16/arc_eval.py`)?

## Method
Plain-language version. Most errors sit in thin bands next to ties between the 2nd and 3rd largest
inputs (and the 1st and 2nd). Pick a tie surface, e.g. x_a = x_b. Every input near that tie can be written as
a point y on the surface plus a step t along the normal d = (e_a - e_b)/sqrt2, with t ~ N(0,1) independent of
y. Along that normal line the network (ReLU, no biases) is exactly piecewise linear, so the set of t where it
is wrong is a union of intervals we can find exactly. Its Gaussian mass is the "signed band width times the
density" from the brief, but computed exactly rather than linearised. The 8+1-dimensional surface integral
of that mass is then estimated from random surface points.

Unbiased decomposition (zones depend on the input only, so the split holds for any weights):

    error = A + B + P_R * rate_R + (11/1024) * rate_blind
    blind = {x_(2) <= 0}                       (probability exactly 11/1024)
    A     = P(wrong, non-blind, x_(2)-x_(3) < gA)            gA = 0.15
    B     = P(wrong, non-blind, x_(1)-x_(2) < gB, not A)     gB = 0.10
    R     = the rest (non-blind, far from both ties)         P_R = 0.569069 by 1-D quadrature

- **A (2nd vs 3rd).** Surface point: a pair (a,b) chosen uniformly from 45, y ~ N(0,I) projected onto
  x_a = x_b, kept only if the tie sits at ranks 2-3 (one other input above, the rest below). That rejection
  uses inputs only. The surface measure 45 * P(ybar in (o_(2), o_(1))) = 45 * 0.100049 is an exact 1-D
  quadrature. On the normal line, inside |t| < min(rank-preserving half-length, gA/sqrt2) and masked to the
  non-blind part (|t| > -sqrt2 * ybar), we:
  1. do one forward pass at the tie (value + slope), which gives the exact affine logits on the whole linear
     piece containing the tie;
  2. walk outward through the misclassified band adjacent to the tie, piece by piece, until the first
     correct point (exact);
  3. integrate the rest of the last walked piece exactly (no extra passes);
  4. estimate any misclassified mass in the still-unknown part of the zone without bias, using one
     Gaussian-weighted random point on the same line (mass x indicator).
- **B (1st vs 2nd).** The same construction on x_(1) = x_(2) (surface measure 45 * 0.060608). The segment is
  clipped to keep r1 and r2 the top two, stay outside zone A and stay non-blind.
- **R and blind.** Plain indicator sampling inside the zone. Inputs are drawn by rejection, which is
  model-free.
- **Allocation.** Two phases. A pilot uses 25% of the budget at default fractions, which came from a pilot
  on the original model only. The rest of the budget is split by a Neyman rule (share proportional to
  weight x s.d. x sqrt(cost)) using that model's own pilot. All samples are used.

Files: `zones.py` (zones, surface sampling, full exact zone walk), `zones2.py` (local band walk + one-point
remainder, cost accounting), `estimator.py` (`est_boundary(model, budget, rng)`), `eval_arc.py`
(harness run), `bias_orig.py`. Earlier steps, kept: `seg.py`/`tie23.py` (full rank-preserving segment walk),
`local.py`/`bands.py` (model-defined bands + remainder; superseded), `tune*.py`, `pilot.py`, `decomp.py`.

**Deductive vs sampling.** The normal-direction integral (the boundary-layer width/mass) is deductive: exact
piecewise-linear root finding. So are the zone probabilities and surface measures (closed-form 1-D
quadratures, 11/1024). The 9-dimensional integral over the tie surface, the one-point remainders, and the
R and blind terms are random sampling. **Overall this is a sampling estimator (conditional Monte Carlo /
Rao-Blackwellisation along mechanistically chosen directions), not a deductive estimate.**

## Status
done. It beats sampling clearly on the original model and at sigma = 0.0025, and only marginally at
sigma = 0.005.

## Results
All numbers count compute in forward-pass equivalents (see Compute accounting). Truths for the noisy copies
are the DRI's `results/leaderboard_truths.json` (4M samples, seed 987654; same `noisy_models(6, sigma, seed=0)`).
24 runs per cell (6 copies x 4 seeds), so each RMSE has roughly +-15% sampling uncertainty.

| model | budget | boundary RMSE | sampling RMSE (analytic, same compute) | MSE gain |
|---|---|---|---|---|
| original (120 reps) | 16k | 0.105 pp | 0.167 pp | 2.5x |
| sigma 0.0025 | 4k | 0.230 pp | 0.386 pp | 2.8x |
| sigma 0.0025 | 16k | 0.124 pp | 0.193 pp | 2.4x |
| sigma 0.0025 | 64k | 0.054 pp | 0.097 pp | 3.2x |
| sigma 0.005 | 4k | 0.433 pp | 0.476 pp | 1.2x |
| sigma 0.005 | 16k | 0.165 pp | 0.238 pp | 2.1x |
| sigma 0.005 | 64k | 0.109 pp | 0.119 pp | 1.2x |

Empirical sampling RMSE from the harness (`est_sampling`, same seeds) is 0.377 / 0.209 / 0.109 pp at sigma 0.0025
and 0.422 / 0.240 / 0.124 pp at sigma 0.005. It is consistent with the analytic values.

**Bias on the original model.** Over 120 runs at 16k the mean is 0.953142 against the truth 0.953179
(20M samples): bias -0.004 +- 0.010 pp. This is consistent with zero. The noisy-copy mean errors are
+0.002 to +0.036 pp, all within about 1.5 s.e.

Commands (run from this folder with the venv active, OMP_NUM_THREADS=1):
- `python bias_orig.py 16000 120` -> `bias_orig_16000.json`
- `python eval_arc.py 0.0025`, `python eval_arc.py 0.005` -> `arc_eval_sigma*.json`
- Zone-width tuning: `python tune2.py` -> `tune2.out` (rejection-sampled surface points, pieces-count cost).
- Per-term unbiasedness checks: `python check8.py 0.1 0.1` (full exact zone walk vs indicator per zone),
  `python check9.py 0.15 0.1` (local walk), `check7.py` (earlier band design vs paired indicator).

Development path (negatives kept):
1. Full exact walk along the rank-preserving 2-3 segment through each sampled x (`tie23.py`). It is
   unbiased, and the variance drops 3.6x vs the indicator, but it takes 15.5 linear pieces per line:
   **4.3x worse** than sampling.
2. Bands defined by the model (the wrong component around the tie) plus a one-point remainder (`local.py`,
   `bands.py`). 0.8-0.9x of sampling. Removing the blind region and adding the 1-2 tie family brought it
   to about 1.5x, but membership checks for the remainder were costly. Superseded.
3. Model-free zones, with the tie surface sampled by its own measure instead of through x
   (`zones.py`). This removes the 1/segment-length weighting. Full exact walk inside the zone takes about
   5 pieces (gates flip right at the tie, because comparator and record neurons switch there): about 2x.
4. Local walk of only the adjacent band, exact use of the known piece, and a one-point remainder for the
   rest of the zone (`zones2.py`): about 3.5x predicted (pieces count) and 2.5-3x measured with full FLOP
   and input accounting.
5. Control variates from pair identity and ybar polynomials explain only R^2 0.08 (A) and 0.24 (B) of
   per-line variance, so they were not used. Stratifying by pair x ybar gained under 10%, also not used.

## Compute accounting
- One full forward pass = 1. A pass carrying value + slope at the tie = 1 + (10-k0)/10, where k0 is the
  first moving input (the prefix before k0 is shared). A continuation pass after a gate flip at step k
  = 2(10-k)/10 (value + slope recomputed from step k). A remainder point on a line = (10-k0)/10.
- Measured per surface line: about 2.9-3.0 equivalents for A and 1.9 for B (about 2.3 / 1.7 linear pieces).
- Input generation: 0.04 per generated 10-vector (about 250 flops vs about 6e3 for a pass). Surface points
  and blind inputs are drawn directly (no rejection, `direct.py`). R-zone inputs are drawn by rejection
  (acceptance 0.57). All of this is included in the reported compute.
- Not counted: the argmax-envelope arithmetic on 10 affine logits per piece (tens of flops), and ndtr calls.
- Model-free precomputation reused across models (free, declared): exact 1-D quadratures for the surface
  measures (45 x 0.100049, 45 x 0.060608), P_R = 0.569069 and the ybar inverse CDFs; P(blind) = 11/1024.
  DEFAULT_FRAC (pilot allocation) comes from a pilot on the original model.
- Leaderboard interface: `estimator.py` exposes `est(model, budget, rng)` and `DEDUCTIVE = False`.

## Claims
- [measured] Each term is unbiased. Zone terms match per-zone indicator means within s.e. (`check8.py`,
  `check9.py`), and the full estimator's bias on the original model is -0.004 +- 0.010 pp over 120 runs.
- [measured] At equal compute (FLOP-equivalent accounting) the MSE is 2.4-3.2x lower than sampling at
  sigma 0.0025, 2.5x lower on the original model, and 1.2-2.1x lower at sigma 0.005.
- [measured] The width of the misclassified band at a tie is highly heterogeneous across the surface:
  per-line mass has CV about 1.2 (median 0.004, 99th percentile 0.042). Simple features (pair, ybar)
  explain under 10% of its variance for A. This intrinsic variance, not the cost of the 1-D integral, is
  what limits the method.
- [inferred] The gain shrinks at sigma 0.005 because those copies make more errors far from ties (R-zone
  error rate up to 8% on one copy vs 0.3% for the original), and that part is plain sampling.
- [inferred] The gate breakpoints cluster at the tie (about 4-5 pieces inside |gap| < 0.07) because the
  record and comparator neurons switch exactly when x_(2) and x_(3) swap.
- [speculative] Exact radial integration (the network is positively homogeneous, so wrong sets are cones and
  the radius integrates in closed form via a t_9 law) would remove only about 5% relative variance. Not
  implemented.

## Alternatives not ruled out
- The gain over sampling may come mostly from **stratification** (separating blind, near-tie and far
  zones, with Neyman allocation) rather than from the exact boundary integral. Partial test: FINDINGS'
  stratified sampler gets 1.2-2.8x. Here, the exact line integral alone (step 4 vs plain sampling inside
  zone A) gives about 14x lower variance per line at about 3x the cost, so the line integral does add
  about 4x within zone A. But the end-to-end gain is similar in size to stratified sampling, and I did not
  run both on identical seeds and budgets.
- The ranking at sigma 0.005 / 64k (1.2x) vs 16k (2.1x) may be noise in the 24-run RMSE: about +-15% on
  RMSE, about +-30% on the MSE ratio. Not resolved with more reps.
- The Neyman allocation re-estimated per model could be what helps on the noisy copies. Fixed default
  fractions were not compared.

## Failure modes
- Models whose errors are not concentrated near ties (larger noise, other tasks): the R term is plain
  sampling and the method falls back to about 1x sampling, plus overhead.
- Small budgets (<= 4k at sigma 0.005): the 25% pilot is noisy, so the allocation can be poor.
- The method needs exact piecewise linearity (ReLU, no biases needed but no other nonlinearity). Lines
  with many gate flips are costly (cap of 60 pieces per side; never hit in tests).
- **Not deductive.** The 9-dimensional surface integral is Monte Carlo. Low-discrepancy/QMC over the
  surface was not tried here; FINDINGS reports no QMC gain for related line-integrated integrands, because
  the per-line mass is discontinuous in the surface coordinates.

