# Exact face-mass tensor compression probe (T29)

## Question
Does exact preintegration of the tie direction leave a low-rank function of the seven lower inputs, including rare wide bands, which tensor cross methods could integrate cheaply?

## Method
`probe.py` freezes two R23 position/value skeletons before observing any outputs: (a,b,max)=(0,1,9), tie quantile .5, maximum conditional quantile .5; and (2,7,4), tie quantile .8, maximum quantile .8. For each of DEV `s0.0025_0` and `s0.02_0`, independently generated 32 row vectors (three lower quantiles) and 32 column vectors (four) define a 32x32 exact-mass matrix. A first 16x16 training block selects pivot rows/columns by pivoted QR. Rank 2/4/8 CUR reconstructs a held-out 16x16 block using only training data and requested cross entries. The whole matrix gives an oracle SVD spectrum; held-out reconstruction is not oracle. Tail error is measured on the top 5% of true held-out masses.

Alternatives recorded before running:
1. Apparent rank decay could reflect a nearly constant ordinary band while the rare peaks remain unrepresented. Cheapest test: held-out top-5% mass recovery and relative tail error.
2. A low-rank fit could merely memorize a finite sampled grid, with unstable interpolation across independent quantile vectors. Cheapest test: training-only pivots and an untouched 16x16 test block.
3. Compression could cost more than plain integration because exact fibers are expensive. Cheapest test: charge all construction cells and QR/SVD/cross arithmetic, separately from diagnostic oracle costs.

This is a sampling-based structural diagnostic, not a submitted ARC estimator. Even if it worked, numerical tensor compression would be unreviewed/quadrature absent a structural mechanistic argument. No post-January-2026 ARC source is used. The generic functional-TT motivation is Gorodetsky, Karaman, Marzouk (2015), arXiv:1510.09088, as supplied by the parent; no paper text was accessed for this probe.

## Status
Done: bounded falsifier, exactly 4096 exact face masses. Low oracle separation rank is present, but held-out cross integration is not uniformly accurate. No estimator or solution claim.

## Results
Four finite matrices have rank-4 oracle energy **98.30–99.60%** and rank-8 energy **99.43–99.95%**. This is better than a high-rank obstruction, but training-only interpolation fails on one predeclared skeleton.

| DEV sigma | Skeleton (a,b,max) | Oracle rank-4 energy | Pivot CUR rank-4 held-out mean error | Stabilized rank-4 mean error | Stabilized rank-8 mean error |
|---|---|---:|---:|---:|---:|
| .0025 | (0,1,9) | 98.78% | +6.99% | +6.20% | +6.75% |
| .0025 | (2,7,4) | 99.60% | +1.29% | +0.11% | -1.02% |
| .02 | (0,1,9) | 98.30% | -9.04% | +1.05% | +0.031% |
| .02 | (2,7,4) | 99.28% | +0.86% | -2.42% | -2.13% |

All mean errors are relative to **conditional held-out face mass**, not percentage-point accuracy errors. `results.json` records predeclared pivot CUR; `matrices.npz` stores all exact masses, cell counts, and row/column quantiles. `stabilize.py` / `stabilized.json` are explicitly post-hoc algebra-only diagnostics: replace the r-by-r pivot inverse with the rank-r truncated SVD pseudoinverse of the 16-by-16 training block, retaining all 16 row/column fibers. Ranks remain 2/4/8, no rank selection or fitted regularization, no additional face evaluations. The choice to stabilize followed observing initial instability, so this is not an independent validation.

Stabilized rank-4 absolute mean discrepancies divided by hypothetical conditional IID-face integration SE at 768 faces are respectively **3.17, 0.048, 0.529, 1.105**; rank-8 values are **3.46, 0.443, 0.016, 0.970**. These are scale comparisons, **not significance tests or ARC MSE ratios**. The Cartesian matrix has dependent cells sharing row/column vectors, its mean is not an IID reference, and the finite-grid conditional mean is not known exactly. The baseline SE is plug-in `std(heldout)/sqrt(768)`; construction and baseline can have unequal costs per face due to different cell counts. A global result would need skeleton integration, independent replicas, exact compute matching, and shared harness evaluation.

Rare-peak check: at sigma .0025 and skeleton (0,1,9), stabilized rank 4/8 misses **20.8% / 27.9%** of top-5% mass in absolute reconstruction error, and has overall relative L2 error **43.2% / 62.4%**. In the spread skeleton, corresponding tail errors are **2.58% / 1.64%**. At sigma .02 they are **18.8% / 13.6%** (early skeleton) and **5.03% / 1.75%** (spread). Top-5% true held-out mass shares are only 11.2–13.1%; these fixed skeletons do **not** by themselves sample the globally rare wide-band configurations. Every exact value integrates all visited cells, with 7.46–12.81 average and 15–30 maximum cells, so this is full mass rather than nearest-boundary width.

Reproduce from repository root:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/tensor_mass/probe.py \
  --checkpoint /workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth
/workspace/clean-neurons/.venv/bin/python m16/approaches/tensor_mass/stabilize.py
```

The initial run fetched the public example through `lb.models(dev=True)`. The optional cached checkpoint loader uses the exact same `lb._noisy(base, 6, sigma, seed=1)` construction and avoids network; checkpoint SHA256 `498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb`. It is used only as a base for DEV copies.

## Compute accounting
Exactly **4096** complete face evaluations and **83,892** value/tangent FPE across all diagnostics; initial run **1.318 CPU seconds** including imports/loading, with ~0.205 seconds inside the four matrix runs. The failed initial restricted network attempt performed no face evaluations. Stabilization adds no faces and about 0.01 CPU seconds of cached linear algebra (interpreter startup excluded). Pivot CUR construction uses 320/384/512 evaluated entries at ranks 2/4/8, including the entire 16x16 pivot-selection block. Rank-4 charges are 5,809–9,837 FPE per skeleton. Stabilization uses **768** entries per case and **11,583–19,563 FPE**, including conservative linear-algebra bounds. Thus stabilization saves only 25% of grid evaluations and does not establish a useful asymptotic algorithm.

The original `sci_tao.core.walk_mass` recomputes a complete value and tangent pass per segment cell. Charge `2*cells` FPE, ignoring its returned optimistic prefix-reuse charge. Construction includes the entire training block plus the requested row and column fibers; held-out values and oracle spectra are diagnostic-only and accounted separately. QR/SVD/pseudoinverse/cross dense arithmetic is bounded explicitly. No training truths or original-model accuracy estimates are consulted. Model-dependent fitting is charged. Per-cell bookkeeping is not fully FLOP-instrumented, so reported FPE are a lower bound; CPU time is reported as an additional check.

## Claims
- [measured] On four predeclared DEV matrices, rank 4 captures 98.30–99.60% Frobenius energy.
- [measured] Neither rank-4 nor rank-8 training-only cross has uniformly small conditional mean error; one skeleton stays 6–7% off after stabilization.
- [inferred] Oracle energy alone is an inadequate success criterion: rare/input-specific residuals and cross instability matter.
- [speculative] Adaptive domain decomposition or regularized functional fibers could retain the observed low rank while controlling the residual. This probe does not establish that possibility.
- [inferred] Low rank on one 3+4 split is necessary for small-rank TT under that ordering, but not sufficient; all TT splits and integration over skeletons must also work.

## Alternatives not ruled out
Different variable ordering, adaptive domain decomposition, different skeletons, and spectral/functional fiber models may have different rank. Four matrices cannot establish generality or global accuracy.

## Failure modes
Rare peaks omitted by cross pivots; ill-conditioned pivot submatrices; error cancellation in estimated integrals; charge dominated by exact line walking; a low-rank value approximation need not yield an accurate global integral; fixed skeletons do not represent the whole face law.
