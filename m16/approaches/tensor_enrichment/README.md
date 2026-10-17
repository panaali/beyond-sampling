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

## Cycle1 results
Controls pass: direct contraction/fiber reconstruction maximum error8.88e-15; the central spike is recovered. A new unqueried spike at(0,2,0,2) remains missed after87/189 queried entries, with true integral.00148809524. This preserves the worst-case support obstruction. Controls cost.01912CPU seconds.

The four frozen neural cases complete in1.544CPU seconds including imports and separately charged full-grid validation. Weighted residual enrichment reduces same-grid absolute accuracy error from.012632/.132418/.273305/.210814(rank2 at H2order3/H2order5/H4order3/H4order5) to.005198/.020928/.004432/.088128. Corresponding rank4 errors are.003382/.015553/.012495/.069905. Thus enrichment improves over its rank2 start, but is not a general improvement over rank4. It uses232/429/231/522 unique faces, versus rank4's310/541/314/489. Weighted and unweighted selection differ only in H2order5; weighted has better full-grid L2 but slightly worse integral error there. No winner is chosen.

Against the actual network reference the H2order3 error worsens from.003809 to−.014020 because the original compression and grid errors cancelled. Other weighted errors are+.017927/+.020774/+.091684, still substantial. Predicted grid values can be negative (down to−.04126); no clipping is applied. Fiber ratios stay below1.60 in weighted runs, so no huge pivot instability explains the remaining errors. These are measured finite-recipe negatives for general accuracy/cost advantage, not a proof against support enrichment.

Reproduce from repository root with OMP/OPENBLAS/MKL threads1, the documented memory limit and shared lock, using `/workspace/clean-neurons/.venv/bin/python -m m16.approaches.tensor_enrichment.check_controls` and `...check_neural`. Full data are `controls_results.json` and `neural_results.json`; every production constructor runs before full-grid validation. Costs remain proxies, not sampling-win certificates.

## Frozen cycle2: local categorical residuals
The remaining alternative is not more global rounds. A configuration index labels distinct error-face geometry; a global product correction extrapolates a residual across this categorical axis. Freeze one rank1 correction independently per configuration, starting from the same rank2 cross and identical central-fiber probes. Select within each configuration by weighted residual magnitude and query only continuous-coordinate fibers; multiply the correction by that configuration's indicator. Integrate with its exact discrete configuration weight. No extra correction rounds, rank search, model/grid change or reference fitting. Retain the two global-round outputs above.

Competing explanations: (a) categorical mixing dominates, so local corrections improve at comparable query cost; (b) within-configuration residual rank/continuous kinks dominate, so local corrections remain inaccurate despite no categorical extrapolation. Test constant/rank2/spike controls and exact fiber/contraction identity first, then all four existing neural cases. Report local rank/term count, queried faces and full construction/algebra cost: one term per configuration is not free rank. Negative predictions remain diagnostic. <=110CPU, same locks and limits; independent review requested. Neural switch-aware probes and local continuous cells remain open, not automatically authorized sweeps.

## Cycle2 local result: mixed, no categorical repair
Local controls pass to3.56e-15; the unseen-spike obstruction remains. On the same H2order3/H2order5/H4order3/H4order5 grids, local same-grid errors are−.031895/+.017404/+.022528/+.039301, versus global weighted−.005198/+.020928/+.004432/+.088128. Thus both order5 cases improve, both order3 cases worsen. Local construction uses280/535/274/603 faces and.05196/.06184/.07048/.09310CPU seconds. It is neither a general categorical repair nor a cheaper universal competitor. The actual-reference errors remain−.040717/+.014404/+.038871/+.042857. Results and full work are in `local_results.json`; controls in `local_controls_results.json`. Reproduce with `python -m m16.approaches.tensor_enrichment.check_local` under the same environment/lock. No best variant is selected.

## Frozen cycle3: additive central-fiber residuals
One distinct inexpensive alternative remains: signed residuals may be poorly represented by products even without huge ratios. Freeze a local additive anchored correction for each configuration using only the already-declared central-fiber probes: for d continuous coordinates and central residual r0, correction is sum_k r(center with coordinate k replaced)−(d−1)r0. This exactly matches each central residual fiber, contains no division, and incurs no additional faces beyond base plus probes. Its discrete integral is the sum of weighted one-coordinate fiber integrals minus(d−1)r0, times configuration weight. No fitting, interaction order increase, resolution change or truth-dependent selection.

Alternative1: multiplicative extrapolation causes the mixed behavior; additive correction improves at lower query cost. Alternative2: unresolved interactions dominate; even the additive rule remains biased. Check independent direct contraction, additive functions with both coefficient signs, constants/zero, central-spike and unqueried-spike behavior, then the same four neural cases. A rank1 product need not be recovered by additive enrichment after an imperfect baseline; do not assert that stronger property. This completes the finite formulation alternatives chosen here; untested neural switch-aware and continuous-local support access remain open. Full construction and diagnostic costs are recorded; <=110CPU, same locks/memory/thread limits.

## Cycle3 additive result and finite alternatives handoff
The additive variant passes exact fiber/independent contraction and signed-additive controls (<=1.78e-15). It deliberately fails the central rank1-spike control: predicted integral−.142857 versus true.005291, despite matching every probed central fiber. This is a concrete unresolved interaction/positivity failure, not a numerical bug. The unqueried-spike obstruction persists.

Neural same-grid errors are−.005099/+.007169/−.030325/−.034770 in the same frozen order. Actual-reference errors are−.013921/+.004169/−.013982/−.031214. Construction uses222/400/218/492 unique faces, fewer than either product correction variant. At order5 it improves both same-grid errors over global products; H4order3 worsens. All settings remain, with no per-model selection. Data are `additive_results.json` and `additive_controls_results.json`; reproduce with `python -m m16.approaches.tensor_enrichment.check_additive` under the same wrapper.

T55 independently checked the real-arithmetic fiber identities and product-weight contraction. Neither identity bounds unqueried error nor guarantees descent. The tested global/local/additive recipes therefore offer limited support-discovery gains but no consistent accuracy/cost winner. Neural switch-aware access, continuous local cells and genuine higher-interaction residuals remain untested; the finite variants do not establish tensor-family impossibility.

### Cost qualification and final bounded falsifier
The saved algebra ledger is PARTIAL: it records prediction, original QR/SVD proxies, fiber normalization and contraction, but does not count every sorting/set/cache operation, residual subtraction, argmax or quadrature-setup scalar operation. Cache requests and constructed probe indices are separately recorded, not converted to exact FLOPs. Unique walker costs are measured proxies; cold CPU includes the full actual algorithm. No complete portable-cost or sampling-win claim is made. T55 identified this qualification before handoff.

Freeze one algebra-only robustness falsifier, not a new neural estimator variant: residual pivot amplitude1e-6/1e-12, three centered fibers whose off-center residual is1, with center weight near1 and off-center weights amplitude^4. The center maximizes weighted squared residual among the probes, but the normalized outer product can explode where multiple coordinates move. Check exact fiber interpolation and expose the resulting off-fiber magnitude. This tests a documented limitation of weighted pivoting; do not silently add clipping, damping or a threshold selected from these outcomes. <=20CPU review lock. The production interface will expose the frozen original weighted two-round/order3 recipe with FIXED_COST and an explicit partial-cost qualification; it is not promoted to a validated solution.

### Final mechanics checks
The frozen weighted-pivot falsifier passes its intended diagnosis: exact probed fibers coexist with off-fiber values1e12/1e24 for pivot amplitudes1e-6/1e-12. The weighted integral remains near the tiny pivot because those off-fiber weights are tiny; this is a stability limitation, not an observed large integral error in the neural cases. `pivot_results.json` records.000458CPU seconds. No correction threshold or clipping was added.

The file-loader-compatible `estimator.py` exposes the original predeclared weighted two-round/order3 recipe as unreviewed, fixed-cost, deductive and COST_IS_PARTIAL_PROXY=True. It reproduces saved H2T4 output.29371046824859826 and partial FPE proxy6140.58 through the shared-style file import. The budget argument does not adapt this fixed recipe; no budget guarantee is made. Other variants remain explicit diagnostic alternatives, not test-selected defaults.

Adapter validation correction: the first temporary validation command omitted `map_location='cpu'` when loading a CUDA-tagged checkpoint and failed before calling the estimator. The preceding adapter statement was committed before that asynchronous failure output was inspected; it was premature at that commit. The corrected CPU load now passes and is saved in `adapter_results.json` (.56940CPU excluding torch import). The initial failed harness CPU was not metered; no neural result file or estimator changed. This retains the failure and correction explicitly.
