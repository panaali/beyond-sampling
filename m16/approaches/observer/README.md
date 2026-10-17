# Psychometric-observer estimator (signal detection theory) for M_16,10 accuracy

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

## Question
Can the accuracy of M_16,10 be computed *deductively* (with no random model inputs) by treating the network as
an observer that compares the top order statistics, where "internal noise" stands for the
deterministic influence of the other 7 inputs? That is: accuracy = closed-form top-3 order-statistic density x
psychometric function P(correct | top-3 values, positions), with the psychometric function built from the weights.
Is it at least as precise as sampling at equal compute (ARC protocol)?

## Method (plain words)
1. **Error decomposition.** acc = 1 - P(b<=0) * blind_err - err23 - err12. Here b is the 2nd-largest input,
   P(b<=0) = 11/1024 exactly, err23 is "picks the rank-3 input" and err12 is "picks the max". Rank-4+ errors are
   not modelled. blind_err comes from `arc_eval._blind_probe` (10 canonical inputs).
2. **Quadrature over the top 3** (`quad.py`). This is a tensor Gauss-Legendre grid over (b, u = b-c, g = a-b), with
   the near-tie gaps clustered, against 720 phi(a)phi(b)phi(c)Phi(c)^7. The 720 position triples are uniform.
   The grid normalisation matches 1-11/1024 to 5e-6.
3. **Anchors** (`v3.py`, `v4.py`). For each position triple, each tie kind (2~3, 1~2), each cell of
   (tie level, other gap, relative near-tie gap r in bins [0,.01,.03,.08,.2,.5,inf]) and each of q Latin-hypercube
   rows, there is one anchor input. Its 7 rest inputs sit at the medians of q equal-mass bins of N(0,1)|<c. The
   exact local linear map at the anchor (`core.map_and_cone`) gives the competitor margin d.x.
4. **Internal noise.** The rest inputs vary inside their bins, and the observer sees this as Gaussian noise on
   the margin. v3 propagates the within-bin variance linearly. v4 (the final version) is *cone-aware*: each rest
   input acts through d_t * clip(x_t, lo_t, hi_t), where [lo_t, hi_t] is the exact interval of x_t on which the
   anchor's ReLU gates (and so its linear map) stay valid. That interval is computed analytically from the
   pre-activation Jacobian. P(err | node) = Phi(-mu/sigma), summed over the quadrature nodes in the anchor's cell.

Files: `core.py` (exact maps, cone intervals, truncated-normal bin moments), `quad.py`, `v0.py`-`v4.py`
(versions), `est.py` (end-to-end estimator and noisy-model evaluation), `estimator.py` (leaderboard entry,
`DEDUCTIVE = True`). Diagnostics (these sample inputs, and are not part of the estimator): `ref.py`, `diag_u.py`,
`diag_cells.py`, `diag_noise.py`, `diag_bands.py`.

## Status
**Done, negative.** The method tracks the noisy copies' accuracy (0.82-0.94) to about 0.6 pp. At equal compute that
is about 10x worse in RMSE (about 100x in MSE) than plain sampling. It does not beat sampling.

## Results
All commands are run from this folder with the venv active. Truth for the original model is 0.953179 (20M samples). For the
noisy copies, `noisy_models(6, sigma)` with 4M-sample truths, reused from `../array_rqmc/results/truth_s*.json`
(same seed and protocol as `arc_eval.true_acc`).

| version | what changed | prediction (original model) | sampled truth | command |
|---|---|---|---|---|
| v0 | one canonical map per triple, global Gaussian over the rest | b>0 error 19.7% | 3.85% | `python v0.py` |
| v1 | anchors at the 2~3 / 1~2 ties (3x3 value cells, K=2) | err23 8.8%, err12 3.7% | 2.86%, 0.72% | `python v1.py 3 3 2 0.02` |
| v2 | + rest stratified in q LHS bins, within-bin noise | q=4: 4.8%, 2.0% | same | `python v2.py 3 3 4 0.02` |
| v3 | + anchors along the near-tie gap axis, no noise | (1,1,2): bias -0.69 pp | | `python est.py 1 1 2 12 0` |
| v3 | same, 2x2 level/gap cells | bias -0.24 pp | | `python est.py 2 2 2 12 0` |
| **v4** | cone-clipped internal noise (1,1,2) | **bias -0.65 pp** (err23 3.69%, err12 1.00%, blind 0.60) | 2.86%, 0.72%, 0.78 | `python est.py 1 1 2 12 1 4 0` |
| v4 | 2x2 / 3x3 cells | -0.35 / -0.28 pp | | `python est.py 2 2 2 12 1 4 0 o` (`3 3 2 ...`) |

ARC protocol, v4 (1,1,2): fixed cost of **215k forward-pass equivalents** per model.

| sigma | per-copy error (pp) | RMSE | sampling RMSE at 215k |
|---|---|---|---|
| 0.0025 | -0.29 -0.33 -0.50 -1.03 -0.42 -0.70 | **0.60 pp** | 0.053 pp |
| 0.005 | +0.14 -0.69 +0.14 -0.77 -0.06 -0.94 | **0.58 pp** | 0.066 pp |

Command: `python est.py 1 1 2 12 1 4 0`. The "evals-eq" printed in `run_v4_1121.log` (85k) uses an older
3-per-anchor cost; the corrected accounting is below. v3 with noise 0 gives a similar RMSE of 0.61 / 0.51 pp
(`run_1121200.log`).

Key diagnostics:
- **Near-tie (relative gap < 2%) predictions are accurate** [measured] (`diag_u.py`). For 2~3 at gaps
  [0,.01) the prediction is 0.0089 against 0.0078 sampled, and at [.01,.02) it is 0.0041 against 0.0048. Before the
  gap-axis anchors, almost all of the excess came from extrapolating one local linear map to large gaps.
- **Internal-noise calibration** [measured] (`diag_noise.py`, 720 triples x 200 rest draws per stratum, at
  relative gaps 0.005 / 0.018 / 0.05 / 0.13). True P(flip) is 0.411 / 0.221 / 0.059 / 0.007. Cone-clipped
  Gaussian: 0.436 / 0.250 / 0.066 / 0.004. Linear Gaussian: 0.439 / 0.296 / 0.141 / 0.058. No noise (point):
  0.439 / 0.235 / 0.067 / 0.003. Per-anchor correlation with truth is about 0.85 for the clipped model. Linear
  "internal noise" extrapolates through ReLU kinks and inflates errors away from the tie. The clipped version removes
  most of that.
- **The remaining err23 excess sits at low tie levels** [measured]. For b < 1 the prediction is 0.0282 against
  0.0177 sampled. For b > 1 it is 0.0086 against 0.0107. Finer level cells reduce the excess slowly
  (-0.65 -> -0.35 -> -0.28 pp), but the cost grows with the number of cells.
- **Quadrature and LHS size have converged** [measured]. nq=20 gives -0.69 pp (vs -0.65), and q=4 gives -0.74 pp.

## Compute accounting
- Each anchor costs 1 forward pass, plus the forward-mode Jacobian of all pre-activations (16x10 per step,
  about 51k FLOPs, about 8.5 passes; needed for the local map and the cone intervals), plus about 1 pass of
  interval work. This is counted as 10.5 passes. v4 (1,1,2) uses 17,280 anchors, so about 181k.
- Quadrature: 2 kinds x 720 triples x q x 1728 nodes x about 40 FLOPs, so about 33k passes. The blind probe is 10 passes.
- **Total: about 215k forward-pass equivalents.** (2,2,2) costs about 760k and (3,3,2) about 1.7M.
- Model-free precomputation is free: tie-level quantiles, grid, and bin medians/moments. Ground truths and all `diag_*` /
  `ref.py` runs use sampling, for diagnosis only.

## Claims
1. [measured] A single linear map per configuration, with Gaussian noise from the other inputs (the
   "canonical point + internal noise" observer), overestimates the error 5x (v0). The map is valid only inside
   its gate cone.
2. [measured] Anchored at near-ties, the exact local map plus within-stratum noise reproduces near-tie flip rates
   to within about 15%.
3. [measured] Treating the other inputs as additive Gaussian noise through the local linear map is miscalibrated
   off the tie (2-8x too many flips). Clipping each input to its exact cone interval fixes most of this at the
   stratum level.
4. [measured] End to end, the deterministic estimator tracks noisy-copy accuracy with RMSE of about 0.6 pp at
   215k pass-equivalents, against about 0.06 pp for sampling. **It does not beat sampling** (about 100x worse in
   MSE).
5. [inferred] Partial cancellation inflates the apparent quality: err23 +0.83 pp and err12 +0.28 pp too high,
   blind region -0.19 pp, rank-4+ errors (about 0.27 pp) not modelled. The component errors are larger than the
   net bias.
6. [speculative] The remaining err23 excess comes from extrapolating the level coordinate (b) inside a cell. A
   homogeneity-aware parametrisation (cells in a/b, c/b, with rest/b distribution shifts) might remove it.

## Alternatives not ruled out
- For claim 6 (the level-extrapolation excess), the alternative explanations are: (a) the LHS rest design with
  q=2 misses gate patterns that are rare but strongly corrective (tested: q=4 does not help, which argues
  against it); (b) the cone-clipping model of rest inputs is wrong in the low-b region, where many rest inputs
  are near 0 and gates flip often (not tested separately at low b); (c) correlated gate changes among several rest
  inputs, which a per-coordinate interval cannot represent (not tested).
- For claim 4, the noisy-copy RMSE could partly reflect the blind-probe and rank-4 omissions varying across copies,
  rather than the observer model itself. Not separated per copy.
- The truths for the noisy copies come from another folder's cache (4M samples, SE about 0.012 pp). That is
  negligible here.

## Failure modes
- Gate cones are pinned at ties. "New 2nd place" and the x9 comparators switch exactly at b = c, so an anchor
  placed exactly at a tie sits on a cone boundary and gives nonsense slopes (seen in v1). Anchors must be offset
  (relative gap > 0).
- Linear extrapolation across ReLU kinks, in any coordinate, produces spurious errors far from ties. That is the
  dominant failure of every "internal noise" variant without cone awareness.
- Clipping the top-3 coordinates to their cone intervals (`clip_top`) is catastrophic (bias -6.5 pp). The
  near-tie variable must be extrapolated linearly.
- Cost scales as triples x cells x LHS rows x about 10 passes. Even the cheapest useful configuration costs more
  than sampling needs for about 0.05 pp. To compete, a deductive method would need bias < 0.05 pp at < 50k
  passes, which this design cannot reach.
