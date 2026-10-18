# T36: reproducing ARC's two M_4,3 estimators (calibrating "roughly match sampling")

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [lit/web_arc](../../../lit/web_arc/README.md) (S4, S5)

## Question

ARC's AlgZoo post says its M_4,3 explanations "roughly match the computational efficiency of sampling". What does
that bar mean numerically? We re-implemented both estimators from Hilton's notes ("Heuristic explanations for
2nd argmax models", 2024-12-21; Robinson, "Symmetric RNNs", 2024-12) and scored them under ARC's protocol:
256 noisy copies, absolute Gaussian weight noise with s.d. 0.05, RMS error, compute in ops. We scored plain
sampling and an exact deductive method the same way. The aim is to calibrate our M_16,10 leaderboard ratio
(MSE / sampling MSE at equal compute).

## Method

- **Model.** `alg_zoo.zoo_2nd_argmax(4,3)`. The default is best-of-5, which is seed 0 here. The weights are cached in
  `m43_weights.npz`. Exact Gaussian accuracy is 0.98455; 10^7 MC samples give 0.98458. This is the model in ARC's notes: the
  weight-sign pattern and the near-symmetries a3 ≈ −a0 etc. all hold, and ARC's figure shows the original at about 0.985.
- **Protocol** (`m43.noisy_models`). We add N(0, 0.05²) to all 32 weights and draw 256 models per noise seed, for noise
  seeds 0-3 (1024 models). The error is estimate − exact accuracy. Ops follow ARC's convention: a multiply-add counts
  as 1, so one forward pass costs 3·4·(4+1)+4·3 = 72 ops.
- **Truth / our deductive method** (`exact.py`). We enumerate linear regions exactly. The model has no biases, so
  every region is a cone in R³ and its Gaussian measure is a solid angle / 4π. We start from the 12 simplicial cones
  {ordering of x} × {sign x0}. We split them by the signs of the step-2 and step-3 pre-activations, using
  Sutherland–Hodgman clipping of spherical polygons. In each final region we clip by the two "correct logit is
  largest" halfspaces and sum the solid angles (fan triangulation, Van Oosterom–Strackee formula, equivalent to
  Girard's formula). One subtle point: where all h3 = 0 the logits tie at 0 and `argmax` returns index 0. That region
  has positive measure (about 0.002 for the base model), and leaving it out biased the truth by +0.0016. Every
  multiply-add, clip, matrix composition and solid angle is counted with `m43.Ops`.
- **Estimator A, "original"** (`est_arc.est_A`). We follow the notes step by step, using each noisy model's own
  weights:
  1. Compute a_j, b_j, c_j. Average the near-antisymmetric pairs into α, β, γ. Form 4 wedge regions in (x0, x1)
     (boundaries γ0/α3, γ1/α2, γ0/β3, γ1/β2), using b-coefficients in the (1,1,0,0) and (1,0,1,0) regions and
     a-coefficients in the other two.
  2. Split by orderings into 3+3+4+4 = 14 likely subregions. The 4 subregions in which x2 lies between x0 and x1 in
     the two thin wedges are "unlikely" and get accuracy 1/3.
  3. In each likely subregion compute p_j = P(z3_j > 0 | S). Replace gate j by a two-point variable driven by a
     latent Z ~ Bern(0.2) that matches mean p and variance p(1−p): for p ≤ ½ it is p − √(q/(1−q))σ at Z=0 and
     p + √((1−q)/q)σ at Z=1, and the mirror image for p > ½.
  4. Compute the accuracy for Z = 0 and Z = 1 as exact 6-halfspace volumes.

  This uses 18 + 56 + 28 = 102 exact volumes, as in the notes. The checks
  max(γ0/α3, γ0/β3) < 1 < min(γ1/α2, γ1/β2) held for all 1024 noisy models.
- **Estimator B, "George's approach"** (`est_arc.est_B`).
  1. Project onto the symmetric subspace by least squares (W_hi = Uᵀw′, W_hh = UᵀW′U, W_oh = W′V). The reduced logits
     are exactly W′_oh|y3(x)|.
  2. Run covariance propagation of ε_z = W_hh ε_h + ε_hh h′ + ε_hi x, with ε_h, h′ and x independent. Every variable
     is zero-mean, and every "multiply by P(positive)" uses ½, as the notes say. This gives
     ε_l ~ N(0, Σ_l) = W_oh ε_h3 + ε_oh h′3, independent of x.
  3. Integrate the 24 six-dimensional pieces (4 sign quadrants × 6 orderings) by sampling (x, ε), as ARC did.
     We use 2^18 samples per model, which adds about 6·10⁻⁴ RMS of noise.
- **Sampling.** The analytic RMSE at k = ops/72 samples is √(mean p(1−p)/k). We checked it empirically at
  k = 16, 107, 986 and 4096 on every noise seed (`run_all.py`).

## Status

Done (2026-10-17). Both estimators reproduce ARC's RMS errors. One deviation is in ARC's sampling comparison
(see below).

## Results

Command: `cd m16/approaches/arc_m43_repro && OMP_NUM_THREADS=1 python run_all.py` (4 noise seeds × 256 models,
about 4 CPU-min). It writes `results.json` and `run_all.log`. The table gives means over noise seeds 0-3, with the
per-seed range in brackets.

| | ARC (notes) | ours |
|---|---|---|
| mean exact accuracy of noisy copies | figure: 0.65-0.97, mostly 0.85-0.95 | 0.881 [0.878-0.885] |
| √mean p(1−p) ("RMS of √(p(1−p))") | **0.24** | **0.320** [0.316-0.323] |
| A RMS error / bias | "around 3%" | **3.29%** [3.08-3.38] / +2.80% (estimates too high, as in ARC's figure) |
| B RMS error / bias | "around 5%", "lower bias" | **4.83%** [4.45-5.14] / +0.88% |
| B with ε dropped (reduced model only) | n/a | 6.79% / +4.8% |
| A ops, ARC's 5k³ rule | 71,000 | 71,040 |
| A ops, measured (our clipping implementation) | n/a | 9,489 |
| B ops, ARC's rule (6-D volumes "by sampling", not charged) | 7,680 | 7,680 |
| exact enumeration ops, measured (mean / max) | n/a | 17,506 / 19,453 (~88 leaf regions), **error 0** |
| sampling RMSE at 71,040 ops (986 samples) | 0.8% | 1.02% |
| sampling RMSE at 7,680 ops (107 samples) | 2.3% | 3.10% |
| sampling RMSE at 9,489 ops (A measured) | n/a | 2.79% |
| sampling RMSE at 17,506 ops (exact) | n/a | 2.05% |

The empirical sampling RMSE matched the formula on every seed. For example, seed 0 gave 0.0351 against a predicted
0.0313 at k = 107, and 0.0093 against 0.0103 at k = 986. The exact truth matched 10⁶-sample MC on all 256 seed-0
models: z mean 0.05, z s.d. 0.94, max |z| 2.4.

**MSE ratio to sampling at equal ops** (our leaderboard's metric; below 1 means better than sampling):

| estimator | ARC's own numbers (0.24) | our numbers, ARC's op count | our numbers, measured op count |
|---|---|---|---|
| A (original) | (3/0.8)² ≈ **14** | (3.29/1.02)² ≈ **10.4** | (3.29/2.79)² ≈ **1.4** |
| B (George) | (5/2.3)² ≈ **4.7** | (4.83/3.10)² ≈ **2.4** | not mechanistic as implemented (its 6-D volumes are sampled) |
| exact enumeration | n/a | n/a | **0** at 17.5k ops. Sampling needs about 7.4M ops (420×) to reach 0.1% RMSE |

ARC's sentence "underperforming sampling by around a factor of 2 in terms of error (i.e, a factor of 4 in terms of
compute)" fits B only. By ARC's own numbers, A is about 3.8× worse in RMSE (14× in MSE).

The q hyperparameter matters for A. Its RMS by q (mean over seeds, same command) is 6.06% at q = 0.05, 4.96% at 0.1,
**3.29% at 0.2**, 2.29% at 0.3 and 4.05% at 0.5. Choosing q = 0.3 on these test models would bring A to MSE ratio
0.67 under the measured count, but that would be advice tuned on the test models.

## Compute accounting

- Ops: a multiply-add counts as 1, and a sqrt, division or atan2 also counts as 1 (`m43.Ops`). A forward pass costs 72.
- Measured counts include every dot product in clipping, every new vertex (with normalisation), every solid-angle
  triangle, and building the per-region linear maps. Branching and list handling are not counted. Ground truth is
  never charged. RNG is not charged for sampling. Both choices follow ARC.
- ARC's count for A uses its "crude" 5k³ per k-halfspace volume and does not count building the maps. Our measured
  count is 7.5× lower, because incremental clipping reuses the 4-halfspace polygon for its 5- and 6-halfspace children.
- Our numbers for B charge nothing for the 2^18-sample 6-D integrals, matching ARC's bookkeeping. If those samples
  were charged (about 30 ops each on the reduced model), B would cost about 8M ops and lose to plain sampling by orders
  of magnitude.
- The whole run takes about 4 CPU-minutes on 1 core.

## Claims

- [measured] Re-implemented from the notes, A gives RMS 3.3% (bias +2.8%) and B gives RMS 4.8% (bias +0.9%) over
  4×256 noisy copies at absolute σ = 0.05. This matches ARC's "around 3%" and "around 5%, although it appears to have
  lower bias".
- [measured] The exact-accuracy distribution of our noisy copies matches ARC's Figure 1 (range about 0.65-0.97). For it,
  √mean p(1−p) = 0.32, not the 0.24 ARC states. ARC's sampling benchmark is therefore about 1.8× too strong in MSE.
  With 0.32, sampling RMSE at ARC's budgets is 1.0% and 3.1%, not 0.8% and 2.3%.
- [measured] Under ARC's op counts, the MSE ratios to sampling are **A ≈ 10, B ≈ 2.4**. With ARC's own (inconsistent)
  0.24 they are A ≈ 14 and B ≈ 4.7.
- [measured] The op-counting convention alone moves A's ratio from 10.4 (5k³ rule) to 1.4 (counted incremental
  clipping). The bookkeeping matters as much as the method.
- [measured] Exact region enumeration of M_4,3 costs about 17.5k ops (about 240 forward passes) and has zero error. On
  M_4,3, a brute-force deductive method beats sampling outright, as ARC did for M_2,2.
- [inferred] B is not mechanistic in ARC's own sense ("avoids any random or pseudorandom sampling"), because its
  6-D volumes were sampled. So the only fully mechanistic bar ARC has demonstrated is A: MSE ratio about 10 by ARC's
  count, or about 1.4 measured.
- [inferred] ARC's 0.24 is most likely an arithmetic slip. The figure itself rules out a different noise model:
  noise on W_hh only, or a lower σ, gives accuracies near 0.93 and A RMS near 2%. See Alternatives.

## What this implies for judging M_16,10 results (calibration verdict)

1. ARC applied "roughly match sampling" to estimators with MSE ratios of **about 2.4 (B) to about 10 (A)** at its own
   op counts, or 4.7 to 14 with its own slipped sampling numbers. The bar of about 4 quoted in our leaderboard
   report (from ARC's "factor of 4 in compute") is ARC's figure for B. The more mechanistic A was about 10× in MSE.
   A leaderboard ratio of at most about 4 therefore meets ARC's demonstrated bar. A ratio of about 10 is still what
   ARC itself called "roughly match". Anything above about 15 is outside what ARC has accepted.
2. Accounting can move a ratio by about 7× (A: 10.4 vs 1.4). M_16,10 ratios are only comparable to ARC's if the
   accounting is stated: FLOPs counted as implemented (our T16/T20 rules), RNG and ground truth uncharged. Our
   leaderboard charges cost as implemented, which is closer to the "measured" column. Under that convention ARC's A
   sits at about 1.4, so the honest-accounting version of ARC's bar is nearer 1.4 than 4.
3. On M_4,3 exact deduction is cheap, about 88 regions. ARC's approximate estimators were a methods exercise, not the
   best deductive option. That regime does not transfer: M_16,10 has vastly more regions (see cone_enum and rank_exact).
4. ARC's noise protocol produced mean accuracy 0.88 from 0.985 on M_4,3. That is a 12-point drop. Our M_16,10 levels
   0.0025-0.005 (relative) give drops of 2-7 points, so they are gentler than ARC's M_4,3 setting (H-E4).

## Alternatives not ruled out

- **ARC's noise draws or truth differ from ours.** Our √mean p(1−p) of 0.32 is stable across seeds (0.316-0.323), and
  the accuracy range matches ARC's figure. Variants tested on 128 models each: noise on W_hh only gives 0.313;
  noise on W_hi + W_hh gives 0.319; σ = 0.03 on all weights gives 0.261 with A RMS 2.0% and B RMS 2.8%. None
  reproduces both 0.24 and ARC's RMS errors, so a slip in ARC's arithmetic is the simplest explanation. The variant
  script was a scratch run, not committed. Its numbers are quoted here only as a check.
- **Implementation choices in A that the notes leave ambiguous.** The OCR of the region inequalities was garbled, so
  we re-derived the boundaries from the stated sign conventions. We assigned "unlikely" to orderings with x2 between
  x0 and x1. We used each noisy model's own a, b, c for the boundaries and the maps. The matching RMS and bias
  suggest these agree with ARC's code, but that is not proven.
- **B's covariance propagation** follows a literal reading: h′ ≈ ½z′, so its covariance is scaled by ¼, and likewise
  for ε_h. Another reading (E[ReLU(z)²] = Var/2) would change Σ_l. B's RMS of 4.8% against ARC's 5% suggests ours is
  close.
- We did not estimate the uncertainty of the RMS from 256 models beyond the seed-to-seed spread (A ±0.15 pp,
  B ±0.35 pp).

## Failure modes

- The exact enumerator assumes no biases (cones through the origin) and uses `argmax` tie-breaking toward index 0
  in the all-dead region. A tie convention that differs from the model's runtime would bias the truth (we hit this
  once: +0.0016).
- The cost of exact enumeration grows with the number of linear regions. That is fine for M_4,3 (about 88) and
  hopeless for M_16,10 without further structure.
- A depends on hand-found structure (sign patterns, near-symmetries, the k-ratio checks). Larger noise would break the
  checks; at σ = 0.05 they held for 1024 of 1024 models.

## Files

`m43.py` holds the protocol, geometry and op counter. `exact.py` holds the exact enumeration. `est_arc.py` holds A
and B. `run_all.py` holds the scoring. `results.json` stores per-model rows and summaries. `run_all.log` is the
printed summary.
