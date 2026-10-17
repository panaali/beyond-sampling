# tube: error mass as integrals over tie faces (thin wedges)

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

## Question
Can the accuracy of M_16,10 (and its noisy copies) be estimated deductively by integrating the misclassified
wedges next to the tie hyperplanes x_a = x_b, instead of probing canonical points, and can that beat random
sampling at equal compute?

## Method (plain words)
1. **An exact way to slice up input space** (model-free). Every input lies on exactly one straight segment
   in a tie direction:
   * if v2 - v3 < v1 - v2 (set R23): hold u = (v2+v3)/2 and all other coordinates fixed and move the
     2nd/3rd pair apart, x_a = u + s/2, x_b = u - s/2. The segment's midpoint is a point on the face
     {x_a = x_b}: one coordinate above, seven below. Along the segment the other ranks never change, and
     the segment stays inside R23 for |s| < min(2(v1-u)/3, 2(u-v4)).
   * otherwise (R12) the same is done with the 1st/2nd pair, with |t| < 2(w-v3)/3.
   * The blind region B = {v2 <= 0} (mass exactly 11/1024) is cut out of both.
   The map (x_a, x_b) -> (u, s) has Jacobian 1 and phi(u+s/2)phi(u-s/2) = phi(u)^2 e^{-s^2/4}, so
   `P(error, not B) = Z23 E_face23[F] + Z12 E_face12[F]`, where F is the Gaussian-weighted length of
   the misclassified part of a segment and Z23 = 360 ∫φ²(1-Φ)Φ⁷, Z12 = 45 ∫φ²Φ⁸ are model-free
   (the density of the v2-v3 and v1-v2 spacings at 0, as in Rényi/David, from 1-D quadrature).
   Check: the face masses reproduce P(R23) and P(R12) to within 4e-4 (`check_mass2.py`).
2. **Homogeneity gives the radial part in closed form.** With no biases the error set is a cone, so the
   face point's radius can be integrated out exactly. F becomes an angular wedge width weighted by
   C ∫(1+κ²/2)^{-5} dκ, with κ = s/|face point| (closed form through ∫cos⁸). This is the Schläfli/Plackett
   angle form of the tube formula.
3. **F per face point is exact, not first-order.** Along a segment the network is exactly piecewise
   linear. The walker runs one forward pass plus one tangent (JVP) pass per linear cell. Inside each cell it
   solves exactly for the slopes where the target's logit loses to some other logit, and adds up the
   weight in closed form. The next cell starts at the first ReLU gate that flips. A cell only recomputes
   from the time step whose gate flipped. (`tube.py: walk_side, F_faces`)
4. **Integration over the face is deterministic.** It uses a fixed-shift Sobol rule in 10 dimensions:
   the face coordinates (u ~ ∝φ²(1-Φ)Φ⁷ by inverse CDF, the max and the 7 lower inputs as truncated
   normals) plus one dimension that picks the positions (a, b, c). Positions are importance-weighted by a
   fixed table q23/q12. Each table entry is proportional to the per-position s.d./√cost of F measured on the
   ORIGINAL model, mixed 80/20 with uniform. The blind region uses a fixed Sobol rule on
   {≤1 positive} with its exact mass. Budget shares: 66% R23 faces, 28% R12 faces, 6% blind.
   (`estimator.py: est_tube`; `est` / `DEDUCTIVE = True` for the leaderboard)

## Status
Done. The estimator is deterministic, uses no random model inputs, and beats sampling by about 2.5-4x in
MSE at equal FLOPs. It is not a closed-form tube formula, though: the first-order formula fails (see Results).

## Results
ARC protocol (`python run_eval.py; python summarize.py`, full output in `summary.txt`). The estimator is
deterministic: one fixed rule per budget, the same nodes for every model. Sampling RMSE is the exact
expected value sqrt(mean p(1-p)/N) at the same compute. Truth s.e. 0.012-0.015 pp.

| models | budget (used) | tube RMSE | sampling RMSE | MSE gain |
|---|---|---|---|---|
| σ=0.0025 ×6 | 1k (945) | 0.455 pp | 0.794 pp | 3.05x |
| σ=0.0025 ×6 | 4k (3952) | 0.233 pp | 0.388 pp | 2.78x |
| σ=0.0025 ×6 | 16k (15995) | 0.110 pp | 0.193 pp | 3.07x |
| σ=0.0025 ×6 | 64k (63385) | 0.050 pp | 0.097 pp | 3.71x |
| σ=0.005 ×6 | 1k (939) | 1.298 pp | 0.982 pp | **0.57x** (one copy off by -2.9 pp) |
| σ=0.005 ×6 | 4k (3949) | 0.325 pp | 0.479 pp | 2.17x |
| σ=0.005 ×6 | 16k (15960) | 0.189 pp | 0.238 pp | 1.58x |
| σ=0.005 ×6 | 64k (63328) | 0.091 pp | 0.120 pp | 1.73x |
| original, 30-40 random shifts of the rule (diagnosis) | 1k / 4k / 16k | 0.429 / 0.215 / 0.087 pp | 0.671 / 0.355 / 0.168 pp | 2.4x / 2.8x / 3.7x |

Original model with the fixed rule: bias +0.012 / +0.232 / +0.104 / +0.007 pp at 1k / 4k / 16k / 64k. These are
single draws of the quadrature error, so they say little on their own. Across 40 shifts at 4k the mean
bias is -0.022 ± 0.034 pp, and across 30 shifts at 16k it is +0.021 ± 0.015 pp: no detectable systematic bias.

Diagnostics on the original model (`diag_*.py`, all short runs):

| finding | number | script |
|---|---|---|
| MC split of the error (4M samples): blind / R23 / R12 | 0.840% / 2.920% / 0.922% | `truth_decomp.py` |
| first-order tube (one linear cell extrapolated over the wedge), R23 term | 0.103 vs exact 0.030; corr 0.20 with exact F | `diag_extrap.py` |
| position of the gate flips along a 2-3 segment (slope κ, median by cell) | 0.010, 0.030, 0.052, 0.075, 0.105 | `diag_cells.py` |
| error mass within κ ≤ 0.02 / 0.04 / 0.08 of the tie (R23) | 85% / 95% / 99% | `diag_split.py` |
| per-face s.d./mean of F (R23), with the radius integrated out | 1.25 (1.28 without) | `diag_rad.py` |
| share of F variance between position triples | 43% | `diag_var.py`, `diag_het.py` |
| cost of an exact walk per face (forward-pass units) | 8.0-9.0 (≈9 cells; 19 without incremental restart) | `diag_cost.py` |
| Newton jump between cells (J=3) | cost 4.4, but misses 3% of the error (multiple crossings) | `diag_newton.py` |
| MSE gain over sampling, 30-40 random shifts of the rule, original model | 2.4x (1k), 2.8x (4k), 3.7x (16k) | `diag_var2.py`, `diag_var3.py` |

Commands: `python run_eval.py` (estimates → `eval_estimates.json`), then `python summarize.py`.
Truth: `arc_eval.true_acc(m, n=4_000_000)` for `noisy_models(6, σ)`, read from `../debiased_series/truth.json`
(the same function and seed; I killed my own duplicate run, see `truth.py`).

## Compute accounting
Unit = one RNN forward pass on one sequence (~6k FLOPs).
- A linear cell costs 2·(10 - t)/10: value plus tangent, recomputed from the time step t whose gate
  flipped. Earlier steps are unchanged; along a segment x varies only at positions a and b. The forward
  prefix before min(a, b) is counted once per face. Root-finding (160 divisions) and the output layer
  (10×16 per cell) are counted as negligible.
- Blind points cost 1 each.
- Model-free precomputation is free: the face-density tables Z and CDF, Sobol points, truncated-normal
  transforms. The importance tables q23/q12 are model-dependent but were computed once on the original
  model (16,384 faces per family, ≈2.8e5 passes), then frozen and reused for every noisy copy. They are
  declared here and are not charged per estimate.
- The `used` column is the measured total, slightly below the budget.

## Claims
- [measured] The co-area slicing is exact: model-free face masses match MC, and walking every segment
  in full reproduces the true error split to within quadrature noise.
- [measured] The first-order thin-wedge (tube) approximation is wrong for this network. ReLU gates flip
  every ~0.02 in slope near a tie, the wedge reaches ~0.05-0.1, and a linearised wedge overestimates the
  R23 error 3.4x with correlation 0.2. Errors are bounded by kinked, curved boundaries, not tilted planes.
- [measured] With exact per-face walks and deterministic face quadrature, RMSE at equal compute is lower
  than sampling's on the original model and on the noisy copies (table above).
- [inferred] The gain is capped at a few x because F varies from face to face (s.d./mean ≈ 1.25 after
  stratifying positions), while each face costs ~8 passes. A Bernoulli sample has variance/mean² ≈ 20 at
  cost 1, so the best ratio is 20/(1.25²·8) ≈ 1.6, raised to ~3 by position importance and the shares.
- [inferred] QMC gives no extra rate: random-shift spread ≈ i.i.d. spread, because F is discontinuous in
  the face coordinates (gate and argmax changes).
- [speculative] Integrating one more face coordinate exactly (a 2-D polygon walk) would lower the CV of F
  but cost more per face. The net effect is unknown.

## Alternatives not ruled out
- The gain could come mainly from *stratification by tie geometry*, which is model-free, rather than
  from the exact walk. Tested in part: `../../stratified.py` gets 1.2-2.8x by stratifying with sampling.
  This method's 2.5-4x is higher, but I did not run stratified sampling with identical shares.
- The good showing on the noisy copies could come from the fixed rule's quadrature error happening to be
  small or correlated across similar models. All copies share the same nodes, and the errors on the σ=0.0025
  copies have the same sign. The random-shift diagnostics (original model only) estimate the typical error.
  I did not run them on noisy copies.
- The importance table is tuned on the original model and could hurt on copies whose errors sit at other
  positions. Not tested separately (no imp=False run on the copies).

## Failure modes
- Cost per face grows with the number of gate flips along a segment. Models with more kinks near ties
  make each face dearer.
- Errors far from any tie (bulk errors) are still integrated exactly, but they inflate the variance of F.
  A model with large bulk-error regions would lose the advantage. On σ=0.005 copies the gain drops to
  1.6-2.2x, and one copy at 1k is off by -2.9 pp. I have not checked whether bulk errors cause this. [inferred]
- The blind region is handled by plain deterministic quadrature: no mechanism there.
- The estimator is deterministic, so a fixed rule has a fixed error for each model. That error behaves like
  a draw from the random-shift spread, not like a shrinking bias.
