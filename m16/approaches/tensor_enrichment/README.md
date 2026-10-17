# T81 weighted residual support enrichment

## Question
Can a small deterministic support probe and residual fiber update repair the reviewed tensor-cross failure at useful cost? Claim e09d04d was pushed before implementation. This folder alone is owned; the existing corrected Python face walker and reviewed tensor implementation are read-only dependencies.

## Method and frozen protocol
Start with the original configuration-first rank2/two-sweep TensorCross. Probe every configuration at the central Gauss node in each continuous coordinate and along every single-coordinate fiber through that center. This is a geometry-independent support check, not random input sampling. On these probes select the residual pivot maximizing its product quadrature weight times squared residual (first lexicographic index breaks ties). Query its complete coordinate fibers and add the rank1 tensor given by the pivot residual times the product of normalized residual fibers. Integrate each product exactly with the discrete product weights. Repeat exactly twice, retaining both intermediate results. No full tensor, SVD or reference accuracy enters the construction.

Freeze comparisons: original rank2, original rank4, and rank2 plus two residual rounds with weighted versus unweighted residual selection. Both enrichment recipes use identical initial probes. New queried fibers join subsequent pivot candidates. A zero residual pivot stops that recipe without dividing by zero; no truth-dependent clipping, rank or stopping selection. Record signed predictions and instability rather than masking them.

First synthetic controls: constant/zero; positive rank2 product sum; known central-fiber rank1 spike; and a fresh spike at an index unqueried on the zero response path. The latter must retain the known impossibility of guaranteed support discovery from partial point queries. Check direct tensor reconstruction versus product contraction and per-fiber residual exactness, including a signed residual. Then the frozen neural cases are H2T4 and H4T4 seed0, Gauss orders3/5, both complete R23/R12 families. Reuse existing qualified reference probabilities .30773074302315223/.47701312599404416. Full-grid evaluations are validation only and are explicitly charged separately; an estimator cannot consult their values. No new MC, noisy copy, larger grid, rank tuning or arbitrary variant follows automatically.

## Alternatives not ruled out
1. Missed support dominates: central fibers expose a residual peak and a rank1 correction improves same-grid integral error. Compare weighted and unweighted residual selection with identical probe geometry.
2. Higher rank/conditioning dominates: two residual products can overcorrect or be unstable even after finding support. Compare original rank4 and report residual-pivot sizes, maximum fiber ratios, signed predictions and full-grid weighted error diagnostically.
3. Coarse quadrature dominates: compression improves while network accuracy does not. Report both errors separately, never select an apparent winner through cancellation.
4. Neural switch-aware probes, local rather than global residual products, higher tensor rank and different continuous coordinates remain untested. No arbitrary point-query scheme can defeat every unqueried spike; this is an empirical rescue under a fixed cost budget.

## Compute accounting
Every unique queried face is counted once per recipe with its corrected walker FPE proxy and cell count; cache lookups, residual predictions, pivot scans, fiber normalization, product contraction, and original QR/SVD proxies are recorded separately. Report cold CPU, source/checkpoint hashes and full diagnostic grid work. These heterogeneous proxies are not a portable scalar-operation certificate; a timing gain is not a sampling win. Construction must not read the full validation tensor. Runs use one thread, <=2GB and <=110CPU under the shared lock; controls <=20CPU may use the review lock. Estimator interface is est(model,budget,rng), with explicit fixed recipe/cost handling once controls pass.

## Status
Protocol frozen; no implementation or run at this commit.

## Results
Pending. Commands and all negatives will be retained here.

## Claims
- [inferred] Existing cross failures separate support discovery from low-rank capacity and grid bias.
- [speculative] Weighted residual fiber enrichment may repair a subset of these failures at limited additional cost.
- No success, exhaustive negative, or full challenge claim is made.

## Failure modes
Tiny pivots, global rank1 extrapolation, sign changes, incomplete probe support, exponential grid validation cost, common walker defects, floating ties, and inaccurate algebra/FPE proxies. No production truth fitting or per-model winner selection.
