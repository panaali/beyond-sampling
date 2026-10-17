# Global tensor-cross face integration (T32)

## Question
Can model-dependent low-rank cross approximation integrate the complete exact error-face laws cheaply enough to compete with sampling?

## Method
Before looking at errors, freeze: TT variable order is discrete configuration then the standard face-law uniform quantiles; all 360 R23 and 45 R12 configurations; Gauss-Legendre orders 3 and 5 for each of nine quantiles; fixed TT rank 2 or 4; two alternating forward/backward QR-pivot sweeps followed by a final forward reconstruction. Pseudoinverse cutoff 1e-10. Initial nested suffix sets are deterministic, spread across the suffix combinations. Integrate each TT core against the product weights, multiply the exact geometry constants Z23/Z12, and return one minus the sum. Every setting runs on DEV sigma .0025_0 and .02_0; no selecting best per model. `estimator.py` provides the standard estimator interface, but classification is unreviewed/quadrature, not mechanistic by determinism.

Predeclared alternatives and falsifiers:
1. Local rank might not survive configuration/top-value variation. Test complete integral and singular values/pivot condition, not just fixed-skeleton fit.
2. Coarse nodes might miss boundary tails even if the discrete tensor compresses. Compare fixed orders 3 and 5 and independent reference; do not attribute all discrepancy to rank.
3. Cross pivots could miss rare high-mass entries. Audit 256 independent model-free grid points per family, including top-5% error, with audit cost separated from construction.

## Status
**Corrected-walker rerun completed:** independent review found a near-coincident-root cache invalidation bug in the shared walker. The exact same eight settings were rerun after backend fix `ae05eef` and local rebuild. Every accuracy, face count, cell count, pivot, and numerical audit metric is unchanged (maximum absolute difference 0); charges increased 12.3–12.6% because new guards are counted. The negative result survives this correction. Original observations remain in `dev_results_before_root_fix.json`; `compare_root_fix.py` generates `root_fix_comparison.json`. No rank, grid, sweep, audit point, or DEV model was changed.

Done: the frozen global TT-cross experiment failed by a wide margin. Eight runs, 40,548 construction face evaluations total, 1.524 CPU seconds for the corrected experiments; each DEV model stayed below the 32,768-face cap. No test-model evaluation and no successful estimator claim.

## Results
Code uses only DEV copies of the public example, built with the exact leaderboard seed1 rule. Existing incremental-walker DEV reference (131072 IID samples) is a low-precision diagnostic, sufficient to establish these large errors but not fine MSE rankings.

| DEV sigma | Nodes/quantile | Rank | Accuracy estimate | Bias vs reference (pp) | Construction faces | Charged FPE |
|---|---:|---:|---:|---:|---:|---:|
| .0025 | 3 | 2 | .990626 | +4.957 | 2,294 | 27,441 |
| .0025 | 3 | 4 | .976884 | +3.583 | 6,374 | 92,686 |
| .0025 | 5 | 2 | .996015 | +5.496 | 2,652 | 71,067 |
| .0025 | 5 | 4 | .973274 | +3.222 | 8,570 | 135,593 |
| .02 | 3 | 2 | .855311 | +23.512 | 2,856 | 46,677 |
| .02 | 3 | 4 | .647425 | +2.723 | 6,014 | 113,261 |
| .02 | 5 | 2 | .799683 | +17.949 | 3,441 | 89,843 |
| .02 | 5 | 4 | .729459 | +10.927 | 8,347 | 158,663 |

References are .9410553 (SE .0006505) and .6201935 (SE .0013406). All reported errors are positive: cross approximation systematically misses error mass. Single squared-error / sampling-variance ratios are 357–12,288. These are diagnostics, not replicated MSE estimates or leaderboard results. Reference uncertainty is much too small to explain the failures.

Audit separates at least part of the error from coarse quadrature. On identical 256 held-out grid indices per family, low-noise true conditional means are R23 .00761/.00891 and R12 .00393/.00446 (orders 3/5). Rank-4 cross predictions average only R23 .00401/.00402 and R12 .00141/.00296. Top-5% absolute mass errors are **67–85%** at rank 4 on low-noise cases, and **16–52%** on high-noise cases; rank 2 is worse. Across both face families, rank-4 relative L2 audit errors are .36–.86. Maximum final-pivot condition number across all runs is below 90; final pivots have the intended rank (last rank is clipped to node count). Thus the problem is not a catastrophic matrix inversion failure.

Weighted audit means imply rough discrete-grid accuracies .9550/.9477 (low-noise, order3/5) and .5326/.6062 (high-noise). These 256-point estimates are noisy and are not exact full-grid integrals. They show that cross compression itself loses mass, while coarse-grid bias is also plausible. Exact full-grid comparison and a global rank spectrum were not computed, so the experiment does not establish an intrinsic impossibility for all TT algorithms.

Synthetic validation in `check.py` reconstructs and integrates exact rank-2 sums of product tensors on three different shapes to absolute error below 1e-10. No neural evaluations were consumed by these checks. The corrected compiled walker is separately validated by T20, including 108 closed-form near-root cases with maximum error 5.56e-17; this task imports it without editing shared code.

Reproduce from repository root:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/clean-neurons/.venv/bin/python m16/approaches/tensor_cross/check.py
TENSOR_CROSS_BACKEND=/workspace/clean-neurons/m16/approaches/sci_tao_v2/compiled.py \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/clean-neurons/.venv/bin/python m16/approaches/tensor_cross/run_dev.py \
  --checkpoint /workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth
```

The run used corrected shared-backend commit `ae05eef` (cherry-picked locally after `f08339a`), built via `python m16/approaches/sci_tao_v2/build.py`. The checkpoint SHA256 is `498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb`. With the compiled walker installed and built in the local repository, omit `TENSOR_CROSS_BACKEND`; otherwise code falls back to Python incremental walking. `dev_results.json` retains all eight runs, audit failures, conditions, and raw accounting. No failed configuration is omitted. The submitted interface defaults to frozen order3/rank2 and ignores budget/RNG (`FIXED_COST=True`); this is a reproducible failed prototype, not a recommended estimator.

## Compute accounting
Corrected-run construction: **40,548 face evaluations**, **735,230.86 FPE** including conservative algebra; 19,890/20,658 faces for low/high-noise models. The 256-per-family audit requested 4,096 values; two were already in construction caches, so additional diagnostic work is **4,094 face evaluations**, **46,156.42 FPE**. The same audit indices are reused across rank settings but evaluations are recharged because caches are local to each independent setting. No construction uses audit observations. Initial exactness tests cost no neural model evaluations. Measured experiment CPU is **1.524 seconds** after imports/checkpoint loading, 0.110–0.305 seconds per run including its audit. No settings were chosen from observed errors. Including the original provisional run and the corrected repeat, this task executed 89,284 face evaluations (81,096 construction + 8,188 additional audit), with about 2.89 experiment CPU seconds total. The old charge counters are preserved for comparison, not treated as corrected accounting.

All unique requested exact face masses count once per setting, including training/pivot search. No cache is shared across ranks/orders/models. New `sci_tao_v2` incremental walker supplies implemented-operation FPE proxies, including roots/CDF/readout bookkeeping. Conservative dense QR/SVD/pseudoinverse/core-product bounds and integration FLOPs count. Geometry grids are model-free precomputation. Diagnostic audit calls count separately and are excluded from estimator cost only because the estimator does not use them. Wall/CPU time reported. No global success claim from two DEV models.

## Claims
- [measured] This rank-2/4, order-3/5, two-sweep TT-cross implementation overestimates accuracy by 2.72–23.51 pp on both predeclared DEV copies.
- [measured] Held-out grid predictions miss high-mass regions despite moderate pivot condition numbers.
- [inferred] The earlier fixed-skeleton low-rank evidence does not extend to useful global integration under this compression scheme.
- [speculative] Greater ranks, weighted residual enrichment or a configuration-aware factorization might repair the failure; this was not demonstrated.
- [inferred] Any generic fixed-grid version remains quadrature unless a further structural argument establishes mechanistic classification.

## Alternatives not ruled out
Different variable ordering, more ranks, continuous adaptive fibers, local decompositions, and weighted residual-driven pivoting could differ. Coarse quadrature and cross error are confounded unless a full-grid reference is feasible.

## Failure modes
Rank-deficient pivot matrices; ill-conditioned interpolation; global tails missed by low-order nodes; overfitting the audit; model-specific accidental cancellation; favorable DEV copies not representative of trained seeds; construction cost overwhelms gain.
