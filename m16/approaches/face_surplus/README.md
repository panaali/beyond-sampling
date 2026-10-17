# Adaptive face-quadrature premise check (T34)

## Question
Does exact integration across the decision boundary leave enough smoothness in the
remaining face coordinates for dimension-adaptive quadrature to be promising?

## Method
Frozen before evaluation: 200 independent diagnostic lines per continuous face
coordinate, both R12/R23 families, H16 T5 seed0 and the original T10 checkpoint,
each perturbed with DEV noise seed1 at relative sigma .0025 and .02 (copy0).
The discrete configuration is held fixed on each line. Each line uses 257 nested
dyadic nodes on [1e-5,1-1e-5]. At level l, its surplus is the odd-node value minus
the average of its adjacent even-node values. Report mean absolute and RMS
surpluses, log2 decay exponents fit over levels4..8, plus interior-only summaries
(coordinate in [.05,.95]) to distinguish tail-transform singularities.

Coordinates are ranked by mean within-line variance, never by accuracy truth.
The backlog's heuristic gate is RMS decay exponent >1.5 in the three leading
coordinates. This is a screening heuristic, not a convergence theorem: mixed
interactions and rare regions may still defeat sparse grids. No accuracy estimator
or parameter fitting is performed in this diagnostic.

Alternatives and separating tests, recorded before evaluation:
1. Apparent irregularity is caused mainly by inverse-Gaussian endpoint behavior:
   compare whole-domain and interior-only decay on the same lines.
2. Typical lines look smooth while rare wide bands dominate: report pooled RMS,
   median line RMS, and top5% line contribution to squared surpluses separately.
3. The effect depends on sequence length or perturbation: freeze the same design
   on T5/T10 and both DEV noise levels, with both face families.

## Status
Done. Negative for the preregistered smoothness screen; no sparse-grid estimator
was built. Protocol/code committed before evaluation (1f98f72). Runs use the
corrected, independently reviewed sci_tao_v2 walker (ae05eef, integrated c68b498).

## Results
The leading three coordinates never all pass the >1.5 RMS-decay screen, using either all nodes or the midpoint-selected interior subset. Exponents below fit levels4..8; an exponent p means
RMS surplus roughly proportional to 2^(-p*l) over these finite levels.

| T | Noise | Face | Leading axes | Whole-domain RMS exponents | Midpoint-interior RMS exponents |
|---|---|---|---|---|---|
| 5 | .0025 | 23 | 1,2,3 | .576,.800,.572 | 1.182,1.073,1.258 |
| 5 | .0025 | 12 | 1,2,4 | .549,.505,.631 | 1.238,1.064,1.191 |
| 5 | .02 | 23 | 2,3,1 | .720,.634,.634 | 1.111,1.466,1.238 |
| 5 | .02 | 12 | 2,1,3 | .724,.609,.750 | 1.118,1.196,1.171 |
| 10 | .0025 | 23 | 1,2,3 | .744,.659,.856 | 1.505,1.584,1.148 |
| 10 | .0025 | 12 | 1,8,9 | .426,.752,.600 | 1.817,1.091,.899 |
| 10 | .02 | 23 | 2,1,8 | .751,.695,.910 | 1.444,1.306,1.264 |
| 10 | .02 | 12 | 7,8,1 | .857,.771,.650 | 1.171,1.128,1.696 |

Axis1 is tie value; axis2 is the maximum for R23 and a lower input for R12.
Other axes are lower inputs in position order. Across all52 coordinate cases,
whole-domain RMS exponents range .426–1.062; interior exponents .885–1.817.
Neither metric clears the screen. An independent reviewer identified an
interpretation limitation: interior selection uses the surplus midpoint. At
level4, all eight midpoints lie inside [.05,.95], but the first and last stencils
still include extreme endpoints. Consequently the interior fit over levels4..8
is not a pure interior-smoothness exponent. A post-hoc sensitivity fit over
levels5..8 lowers every leading-three interior exponent to .703–1.144; the
negative screening decision is unchanged. No model evaluations were repeated; values are saved in `sensitivity.json`.
Across all52 axes that sensitivity range is .703–1.292.
At level8, the top5% of diagnostic lines carry 40.4–95.0% of interior squared
surplus in the leading coordinates: mean smoothness hides substantial concentration.
These figures concern surplus, not the share of the network's error probability.

Raw per-level values are in `results.json`; `run.log` contains the run output.
Reproduce from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python m16/approaches/face_surplus/probe.py --checkpoint5 work/checkpoints/H16_T5_seed0.pth --checkpoint10 m16/approaches/mechanism_checks/checkpoint.pth
```

T5 is explicitly **seed0**, not the zoo's best-of-five seed2 used by some other
studies. Therefore this is not a matched replication of their T5 numbers. No
accuracy truth was used. Public checkpoint:
https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_16_5_1073741824_0.pth
SHA256 `c026d6be1a44e4004d009443b768411e046e75ef42090e525b5ac8a3719e7040`.
T10 checkpoint SHA256
`498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb`.
Independent synthetic algebra check: F(u)=u² gives all four fitted exponents2
within1e-12; a single fixed hinge gives whole-domain L1≈2.040/RMS≈1.540.

## Compute accounting
2,672,800 face evaluations, 30,073,718.76 returned walker FPE, 27.04 CPU seconds
on one thread, excluding interpreter/import/checkpoint loading. All gate cells
and the walker's root/readout/CDF bookkeeping are included. Face construction,
model-free tables, perturbation generation and surplus statistics are not
instrumented in this FPE total; it is not an end-to-end efficiency measurement.
The entire run is diagnostic overhead, not an estimator cost. The frozen
300 CPU-second cap was not reached.

## Claims
- [measured] None of eight model/family cases meets the frozen leading-three
  coordinate screen, even using interior-only decay.
- [measured] The midpoint-interior and whole-domain finite-level statistics differ
  substantially, with a stencil-selection confound in the fitted decay rates.
- [inferred] These results do not justify building the proposed generic sparse
  grid in the current coordinates without an additional structural improvement.
- [speculative] Boundary-aware or transformed-coordinate integration could still
  help. These data neither prove impossibility nor identify a T5→T10 transition.

## Alternatives not ruled out
Sparse grids in a different coordinate system, model-dependent coordinate changes,
weighted quadrature, and boundary-aware adaptive meshes are not tested.

## Failure modes
Finite levels cannot establish asymptotic rates; random diagnostic lines may miss
rare important regions. One checkpoint per length does not establish universality.
Coordinate rankings use variance on a finite equally spaced grid, not exact Sobol
indices. One-dimensional smoothness alone does not bound mixed derivatives. The same
200 anchors are reused across coordinates and noise levels within each length
and family; these are not independent sets of anchors for every case.
Numerically exact walking is not interval-arithmetic certification. Endpoints are
clipped and cannot characterize the omitted extreme tails.
