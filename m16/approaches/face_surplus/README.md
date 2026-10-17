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
Protocol frozen; awaiting the independently checked segment-walker correction.

## Results
Pending. Reproduce from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python m16/approaches/face_surplus/probe.py --checkpoint5 work/checkpoints/H16_T5_seed0.pth --checkpoint10 m16/approaches/mechanism_checks/checkpoint.pth
```

## Compute accounting
All face evaluations and returned walker FLOP proxies are reported. Model-free
face construction costs are reported separately using the shared table convention.
There are 2,672,800 face evaluations at the frozen setting, each including all
gate cells. This is diagnostic overhead, not an estimator cost. One thread,
300 CPU-second cap; save each completed coordinate before checking the cap.

## Claims
[speculative] Useful low-dimensional smoothness survives exact preintegration.

## Alternatives not ruled out
Sparse grids in a different coordinate system, model-dependent coordinate changes,
weighted quadrature, and boundary-aware adaptive meshes are not tested.

## Failure modes
Finite levels cannot establish asymptotic rates; random diagnostic lines may miss
rare important regions. One checkpoint per length does not establish universality.
Numerically exact walking is not interval-arithmetic certification. Endpoints are
clipped and cannot characterize the omitted extreme tails.
