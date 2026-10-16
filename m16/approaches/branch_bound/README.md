# Probabilistic branch and bound for M_16,10 accuracy (certified L <= acc <= U)

## Question
Can the accuracy of M_16,10 be bounded *deductively* from the weights, with certified lower and upper bounds,
by partitioning input space into regions of exactly known Gaussian measure and proving each region
"always correct" or "always wrong" with linear bound propagation (Boetius, Leue & Sutter, ICML 2025)? How does the
bound gap U - L shrink with compute, compared with Monte Carlo?

## Status
**Done. Negative result.** At 1.05e11 FLOPs (17.4M RNN-pass equivalents, ~61k bounded cells) the certified
mass is **0**: L = 0, U = 1 - 5.5e-7. Probes of the cell size needed for certification put a 20% certified
fraction at ~5e10 cells (~1e16 FLOPs). Monte Carlo reaches +-0.005 pp at that same 1e11 FLOPs.

## Method (deductive, no random model inputs)
* **Regions (`cone.py`).** Each region is a root (a, b, s): x_a is the max, x_b the 2nd (so the target b is fixed),
  and s = sign(x_b). That gives 180 roots. The net is positively homogeneous, so I work with cones, normalised by
  lam = |x_b| + x_a - x_b. The chart coordinates are w = (theta, eps_1..eps_8). For s=+:
  x_a=1, x_b=theta, x_j=theta-eps_j. For s=-: x_a=1-2theta, x_b=-theta, x_j=-theta-eps_j.
  Every ordering constraint of the root is a box face, so a region is the cone over an axis-aligned box in w.
* **Exact, model-free measure.** Given (lam, theta), the 8 other inputs are independent. So the measure of a cone over
  a box is a 2-D integral of products of Phi, computed by deterministic Gauss-Legendre quadrature (40 x 12-24 nodes).
  Checked three ways: the roots sum to 1 - 3.1e-5 (that is the truncation eps <= 64, counted as undecided); splits
  are additive to 1e-12; it agrees with a model-free MC check on one box (0.001191 vs 0.001194).
  Children's masses are set to sum exactly to the parent's.
* **Bounds (`core.py`).** The RNN is unrolled into 10 layers of 16 neurons. Intermediate pre-activation bounds use
  forward symbolic propagation tightened by per-layer backward CROWN. The 9 margins logit_b - logit_j and their
  negatives use backward CROWN. A region is certified correct if every margin's lower bound is > 0, and certified
  wrong if some margin's upper bound is < 0. Soundness was checked on 2,000 boxes with 0 violations.
* **Branching (`bb2.py`).** Best-first by undecided mass, in batches. I bisect the coordinate with the largest
  |coef| x width in the worst margin's CROWN bound. Wide eps ranges are split geometrically.
* **Point estimates.** Three rules: (i) midpoint (L+U)/2; (ii) "centre": L plus the mass of undecided cells whose
  representative point the net classifies correctly, which is a deterministic cubature; (iii) "linear share": the
  worst margin's lb/ub interpolation.
* `bb.py` / the first part of `core.py` is the earlier, non-homogeneous version (roots (a,b), absolute gap
  coordinates, 1-D measure integral). It also certified nothing after 5M RNN-equivalents.

## Results
Truth for the original model: 0.953179. Truths for noisy_models(3, 0.0025) at n=4M (`truth.py`, `truth.json`):
0.934801, 0.915305, 0.940846.

**1. Bound gap vs compute, original model** (`python bb2.py crown 1e11`; log in `results_orig_run.log`)

| FLOPs | RNN-equiv | open cells | L | U | centre est.* | linear-share est. |
|---|---|---|---|---|---|---|
| 9.6e8 | 1.6e5 | 360 | 0 | 1 | 0.815 | 0.466 |
| 9.7e9 | 1.6e6 | 2,880 | 0 | 1 | 0.858 | 0.475 |
| 4.8e10 | 8.0e6 | 13,952 | 0 | 1 | 0.780 | 0.471 |
| 1.05e11 | 1.74e7 | 30,335 | 0 | 1 - 5.5e-7 | 0.792 | 0.468 |

\*This run used the plain box centre. `estimator.py` uses a geometric representative point for wide eps ranges.
Sampling at 1.74e7 passes has RMSE 0.005 pp. The bound gap stayed at 100%; the point estimates are off by 16 pp
or more.

**2. Noisy copies, sigma=0.0025** (`python eval_noisy.py 1e10`; `results_noisy0025_1e+10.json`, `run_noisy.log`):
see the section "Noisy-copy run" at the end. L = 0 on all three copies.

**3. Leaderboard estimator** (`estimator.py`, DEDUCTIVE=True). Estimates at budgets of 1k and 256k passes for
(orig, noisy0, noisy1, noisy2) were 0.989/0.978/0.881/0.989 and 0.989/0.978/0.891/0.989. That is an error of
+3.6, +4.3, -2.4 to -3.4, and +4.9 pp. Sampling's RMSE is 0.7 pp at 1k and 0.04 pp at 256k.

**4. Why: how small must a certified cell be?** These are diagnostics, not part of the estimator, and use random
centres.
* `scaling_probe.py` (cone cells of half-width h around 1,500 Gaussian draws; N_tile = median of
  1/cell mass is roughly the number of such cells needed to tile the mass):

  | h | CROWN-certified correct | median N_tile |
  |---|---|---|
  | 0.3 | 0.2% | 3.7e6 |
  | 0.1 | 19% | 4.9e10 |
  | 0.03 | 76% | 2.2e15 |
  | 0.01 | 90% | 4.3e19 |

  With *anisotropic* cells sized by the exact local margin gradient (an idealised adaptive split), certification
  reached 0.3 / 1.5 / 4.8 / 14% at median N_tile 3e4 / 6e5 / 2e7 / 1.4e10.
* `intrinsic_probe.py` (`intrinsic_probe.log`) separates relaxation looseness from real non-linearity. At each h it
  compares the cells CROWN certifies with the cells that are truly all-correct (sampled), as a stand-in for a
  perfect complete verifier:

  | h | CROWN-certified | truly all-correct | median gates (neuron, step) varying in cell |
  |---|---|---|---|
  | 0.1 | 20% | 65% | 14 |
  | 0.03 | 75% | 87% | 4 |
  | 0.01 | 91% | 92% | 1 |

  This agrees with the DRI's count: 535k distinct gate patterns in 1M samples, and errors spread over 38.7k
  patterns.
* `poset_probe.py`: in absolute gap coordinates and at a fixed box width, fixing more of the order (a top-k chain
  instead of the (a,b) star) does not raise the certified fraction (k=2: 82%, k=3: 82%, k=5: 81%, k=10: 65% at
  half-width 0.03). Ordering chambers do not buy larger certifiable cells.

## Compute accounting
* One bounded cell costs about 1.6e6 FLOPs (~270 RNN passes): 2.4e5 forward-symbolic, 1.2e6 per-layer CROWN,
  1.7e5 final CROWN, and about 2e5 for the measure quadrature. That last part is model-free, but I counted it.
  Constants are in `core.py`.
* The centre rule adds one forward pass per open cell at each snapshot.
* Model-free work: Gauss-Legendre nodes. Ground-truth sampling is evaluation only.
* Measured wall-clock throughput on the shared CPU (load ~40) was about 100-150 cells/s.

## Claims
* [measured] Certified probabilistic B&B with CROWN-style relaxations certifies **0** mass of M_16,10 within
  1.05e11 FLOPs (17M RNN-equivalents). The gap U - L stays at 1. Sampling at that compute has RMSE 0.005 pp.
* [measured] Certification needs cells of relative half-width <= 0.03-0.1 in all 9 normalised coordinates. Even a
  perfect verifier certifies only 65% of h=0.1 cells, because a median of 14 gate states change inside such a cell.
* [inferred] A 20% certified fraction needs ~1e10-5e10 cells, about 1e16 FLOPs. A gap below 10% needs more than
  1e15 cells. B&B therefore loses to sampling at every feasible precision, by many orders of magnitude.
* [inferred] The cause is the number of linear regions (combinatorial gating), not loose relaxations. At h=0.03,
  CROWN already gets 75% of the 87% that is truly certifiable.
* [measured] Deterministic cubature on the B&B cells (centre rule) is biased by +2 to +5 pp at 1k-256k passes,
  and by -16 pp at 1e11 FLOPs with plain centres. It is far worse than sampling.

## Alternatives not ruled out
1. *A better verifier* (alpha-CROWN, ReLU-split BaB, exact MILP per cell) could close the 20% vs 65% gap at h=0.1.
   Not tested. The intrinsic probe bounds the possible gain: even perfect verification needs ~1e10 cells for 65% of
   h=0.1 cells, so this cannot change the conclusion.
2. *Better cell geometry*: cells aligned with activation-pattern boundaries, or ordering refinements combined with
   anisotropic splits. Partly tested: idealised gradient-shaped anisotropic cells did not help, and fixing the order
   (poset_probe) did not help. Non-box polyhedral cells would need orthant-probability integrals for their measure;
   I did not try them.
3. *A smarter branching or priority rule* could get L > 0 at 1e11 FLOPs. Probable, but by the tiling numbers it
   would stay at a few percent.
4. The probes use random cell centres, which is diagnostic only. Their N_tile figures are rough (median of
   1/mass) and could be off by 1-2 orders of magnitude. The conclusion does not depend on that.

## Failure modes
* Ties: every root cone contains tie points (x_b = x_j, or x_a = x_b) where the target switches. Those
  neighbourhoods can never be certified, so the gap has a floor set by the cell size near the tie surfaces.
* Truncation eps <= 64 leaves 3.1e-5 of the mass permanently undecided.
* The measure comes from quadrature, which is deterministic and accurate to ~1e-12 relative on tested boxes, but
  is not an interval-certified integral.
* Blowing up the bound: rho(|Wh|) = 3.76, so plain IBP widens bounds by about 5e5x over 10 steps. Only symbolic or
  CROWN bounds are usable.
