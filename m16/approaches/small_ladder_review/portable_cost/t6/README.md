# T52 counted independent H2T6 DD geometry

## Question
Can independent DD geometry reduce the H2T6 simplex count enough to support a fully counted deductive integration comparison?

## Method
Parent pushed extension b7209a1 before this work. Edit only portable_cost/t6; preserve T4/T5. Begin with base seed0 geometry only. Read the checkpoint directly and derive gates, first-index output regions, second-argmax regions, orthant cuts and pulling triangulation. Do not use author ray caches or select regions from author results. Reuse the audited DD wrappers with width6 bounded rank and scalar6x6 determinants.

Constraint IDs are explicitly disjoint: gates0..66 for H2T6, output80..85, target-rank86..91, orthants100..105, all within two64-bit words. T5's hard-coded output IDs are unsuitable here. Record every ambiguity/failure with partial work and exact failing matrix. No opaque numerical kernel in production; any SVD diagnostic is separate.

## Status
Frozen protocol before implementation. First produce complete gate and correct-region geometry with full scalar ledger, CPU, simplex counts and deterministic affine checks. The orthant L1 cross-section determinant sum of the full gate partition must equal64 within declared floating diagnostic tolerance. This is an empirical partition check, not an exact geometry certificate. Only after reporting these results will triangle integration be planned.

## Results
Pending. Frozen external comparison only: author Qhull route has136438 correct simplices and accuracy0.24815022292589073 with conditional bound9.04235e-6. Do not use that accuracy to guide geometric decisions or drop regions.

## Compute accounting
Charge all gate/output/rank/orthant clipping, adjacency tests, face traversal, width6 bounded rank, normalization and6x6 determinants. Report gate-partition validation work separately from production correct-region work. Scalar arithmetic/Boolean/64-bit bitset operations cost1; transcendental cost20. Memory copies/addressing, containers and interpreter overhead remain excluded from the arithmetic metric but included in CPU. Save geometry and source hashes. One thread,2GB memory, <=110CPU per cycle, common compute lock for potentially long geometry. No random model inputs and no numerical mass integral in this phase.

## Claims
[speculative] DD may substantially reduce author Qhull triangulation size. [inferred] This geometry pipeline preserves the tested T4/T5 structure but needs dimension6 validation. No efficiency or solved claim yet.

## Alternatives not ruled out
1. A smaller count might omit or duplicate regions through constraint collisions or rank decisions. Falsifier: explicit ID-disjointness assertions, full gate determinant sum64, retained rank ambiguity and shape/count diagnostics.
2. Width5 counters/routines might silently undercount or mis-handle width6. Falsifier: scalar determinant and rank primitives tested on fixed6D fixtures, complete operation components saved, and no unsupported width accepted.
3. Floating tolerances may corrupt narrow cones. Falsifier: save any ambiguous matrix, compare separately with SVD, retain failures; partition and deterministic gate-map checks provide additional evidence but not a full roundoff certificate.

## Failure modes
Floating DD tolerances and rank brackets, combinatorial growth, finite bitset IDs, long exact determinant sums, and eventual triangle-integration setup/roundoff cost. Formal geometry certification remains a separate T68 workstream. No author cache reuse, region omissions or silent rank fallbacks.

### Geometry cycle results
Width6 primitive checks passed:20 scalar determinants agree with NumPy within1.43e-14 absolute error, full-rank/rank5/threshold-ambiguity fixtures pass, and all four constraint-ID sets are disjoint. Primitive CPU0.0069 seconds.

Independent base DD geometry completed with233 gate cells (stage counts2,5,13,34,89,233),42,012 correct-region simplices and2,469 gate-partition simplices. The correct count is30.8% of the author's136,438 Qhull simplices; no author geometry cache was read. Full partition determinant sum is64.00000000000009, correct determinant sum15.808210496355025, and deterministic gate affine-map error is4.44e-16. No rank ambiguity, lower-dimensional side omission or zero-determinant simplex occurred. There were70,186 production bounded-rank calls and3,428 partition-validation calls, with no SVD.

Complete geometry/checkpoint cycle took14.051 CPU seconds, of which12.957 was production geometry and0.939 partition/affine validation. Production scalar operation upper bound is684,419,366; separate validation bound32,973,052. The fixed500-operation root setup allowance includes construction/checking of the dimension6 constraint-ID sets and the initial gate state; DD's128-entry bitset table is separately charged. The saved ray cache is independently generated, ignored by Git, and hashed in `geometry_results.json`; numerical mass integration has not yet run. Floating topology remains empirically checked, not certified.

Reproduce under the declared memory/thread caps:
```
flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python -m m16.approaches.small_ladder_review.portable_cost.t6.check_primitives
flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python -m m16.approaches.small_ladder_review.portable_cost.t6.run_geometry
```

### Frozen integration-planning cycle
Before any mass evaluation, build counted scalar Jacobi rules for outer weights(1-u)^4,(1-u)^3,(1-u)^2. Use explicit tridiagonal Jacobi coefficients,60-step Sturm bisection for each node, and Christoffel weights from the polynomial recurrence; no opaque eigensolver. Validate polynomial moments through degree2n-1 and compare SciPy nodes/weights only as separately reported reference checks. This allows Chebyshev degree K=n-1 after exact inner triangle integration, whereas ordinary GL would allow only n-3.

Plan orders4,6,8,10,12,16,20,24,28,32 from the closed q^-3 Chebyshev tail reviewed independently by root/T48, using the exact same positive-quadrature factor2*det/(120*pi^3). Geometry gives radial bounds from1/6, minimum Gram entry and centroid supporting plane; add explicit outward numerical cushions but do not call them interval certification. Targets1e-5/1e-6/1e-7 divide equally among the42012 independently generated cones; retain any unresolved bound. Charge all radial setup, determinants, order trials and rule construction. No triangle integral runs in this cycle.

Falsifiers: an incorrect Jacobi normalization or recurrence fails monomial moments and independent node/weight comparison; a wrong degree or tail factor fails identity-cone/elementary coefficient algebra checks. Strict common-orthant and L1 checks on the independently generated rays must pass before invoking T55's triangle primitive. Planned triangle/node count alone is not an operation cost: later work must charge the full analytic triangle path and positive fallback including its rule construction.

### Planning diagnostics and explicit projection protocol
The first Jacobi implementation failed the alpha2 moment test: an exact zero Sturm pivot was counted before its negative limiting sign was installed. Counting the guarded pivot instead fixes it. All15 rules (orders4/8/16/24/32, alpha2/3/4) now agree with SciPy nodes within1.12e-16 and weights within3.02e-15; every weighted monomial through degree2n-1 agrees within1.67e-16. Validation CPU0.0612 seconds; SciPy was reference-only.

The strict orthant falsifier found10545 DD cones with opposite tiny coordinate signs. The largest wrong-sign coordinate is1.582e-16; projecting columns onto the common orthant selected by each coordinate's largest-magnitude component, then L1 renormalizing, shifts any column by at most4.447e-16. The real-arithmetic Gaussian mass perturbation bound below sums to5.325e-12 using a simple norm<=1 estimate. This diagnosis took0.0954 CPU seconds and evaluated no mass. Original rays remain preserved.

Freeze the explicit correction before planning: save a separate projected-ray cache, charge2048 scalar-equivalent operations per cone for projection, normalization, vector-error calculation and its bound, and add the summed perturbation bound to all eventual error budgets. Let epsilon_j=||r_j-s_j||, epsilon=max_j epsilon_j, M bound all original/projected column norms, and a=min_j||s_j||_1/sqrt6-epsilon>0. Multilinearity/Hadamard gives |detR-detS|<=M^5*sum epsilon_j. Along the connecting simplex points, ||z||>=a and the gradient norm of||z||^-6 is<=6*a^-7. Therefore the per-cone mass shift is at most
`[M^5*(sum epsilon_j)*a^-6 + M^6*6*epsilon*a^-7]/(120*pi^3)`.
This does not require projected cones to remain disjoint: the sum of individual integral shifts is bounded. Any collapsed cone must be recorded, not silently omitted. Computed errors/norms are still ordinary floating arithmetic, so this is an explicit analytic perturbation allowance, not a complete interval certificate for clipping or arithmetic. No triangle integration until this corrected planning pass is reported.

### Corrected planning results
Explicit projected rays have no mixed orthants and no collapsed determinants. Using measured column norms in the Hadamard bound tightens the summed projection allowance to3.1396e-12. Projection and its bound cost86,040,676 scalar-equivalent operations, fully charged in subsequent totals. The original cache and the separate projected cache are both retained and hashed.

Planning targets1e-5/1e-6/1e-7 give actual outer bounds2.453e-6/2.687e-7/2.811e-8 and54,899,328 /93,181,984 /145,752,360 planned triangle evaluations. The strictest plan exceeds22 individual cone allocations at the maximum order32, but its summed bound still meets the global1e-7 target; no cone is omitted. Full geometry+projection+planning+cold outer-rule setup costs866,906,835 /875,013,555 /878,501,595 scalar operations, before triangle evaluations. The planning cycle took3.708 CPU seconds and evaluated no mass. `planning_results.json` saves complete order histograms, counts, bounds and costs.

T55's batch triangle primitive contains no SVD/QR/linear solve. Its frame/norm/area/degree3-moment route needs a conservative scalar counter, while its positive fallback must receive counted rule construction rather than NumPy's hidden leggauss setup. The proposed next bounded cycle replaces only that local fallback, validates primitives against T55 controls, and runs a small fixed deterministic subset to test cost/stability before any full93.2M-triangle calculation. The entire eventual mass calculation must include the projection, outer quadrature, inner positive fallback and analytic series allowances separately.

### Frozen counted triangle primitive cycle
Load the T55 batch source read-only, replacing only its imported positive fallback in the local namespace. Restrict to ambient dimension6. Charge a deliberately loose10,000 scalar-equivalent operations per attempted triangle plus10,000 per batch for all frame calculations, norms, cross products, degree3 moment algebra, powers, trigonometric functions, screens and metadata reductions. This is a source-level arithmetic upper allowance, not an instruction count: fewer than1000 ordinary operations and100 special-function equivalents are present per triangle, leaving substantial slack under TRANS20.

The local fallback uses the same radial bounds and positive2D Duffy rule, but replaces the p=3 gamma ratio by(n+1)(n+2)/2 and constructs cached GL rules with the counted scalar Newton method. Charge512 for radial setup,120 per candidate-order bound,100 operations per positive quadrature node and all cold GL/Duffy setup. No opaque eigensolver, QR or SVD may occur. Falsifiers are the15 existing T55 six-dimensional width/degeneracy controls, the exact constant-triangle value, and high-order GL moment checks. No trained-model mass evaluation in this primitive cycle.

### Counted triangle primitive results
All15 existing six-dimensional T55 width/degeneracy controls pass the frozen comparison to saved positive references; maximum relative difference5.14e-11 reflects the inherited analytic route's qualified floating screens. Seven cases use the fully counted positive fallback. The exact constant triangle at coordinates1/6 returns108 as expected. Cold scalar GL rules at orders48/64/96/128 integrate all monomials through degree2n-1 within2e-14. Primitive/control CPU0.0221 seconds; no trained-model mass was evaluated.

The local source adapter pins T55 batch SHAed72ccfb52ee6f43afc19384dcc7f164f5022dc7c2610ba94da52f58b8e81461 and refuses an unaudited source revision. It leaves the shared source unchanged. Both analytic and fallback paths report full declared scalar costs, with no opaque setup kernels. Analytic-series and positive-quadrature bounds remain separate from the heuristic roundoff screen; no fully rigorous floating bound is claimed.

### Frozen bounded pilot
First check the full six-dimensional identity cone at outer Jacobi orders8 and16 against mass1/64. Then evaluate exactly64 equally spaced deterministic cone indices from the independently generated projected cache, using the frozen1e-6 plan and a global inner-quadrature budget1e-8 divided over all42012 cones. This pilot is a partial mass only; never scale it into an accuracy estimate. Save exact indices, all bounds/counts, cold outer/inner setup, and every failure. Runtime guard95CPU under the primary lock. Falsifiers are incorrect triangle-to-cone normalization on the identity and unstable/expensive fallback behavior on the fixed subset. Full-model integration remains a separate checkpointed cycle after this pilot report.

### Bounded pilot results
Both identity-cone checks satisfy their analytic outer/inner bounds around1/64. All64 frozen model-cone indices completed with no failure in0.632 CPU seconds (0.681 including identity controls). The partial mass0.0030881937704798613 is not a full accuracy estimate. Work was167,352 triangle evaluations,3,641 positive fallbacks and58,256 positive-quadrature nodes. Fully charged pilot arithmetic is1,711,389,234 operations including cold rule setup; no opaque kernel was called. Weighted inner error allowance is6.453e-19 and analytic-series allowance7.108e-27, separately from the partial outer/projection allowances and uncertified roundoff.

The pilot supports proceeding to a checkpointed full integral under the same frozen plan and kernel source. Extrapolated runtime/cost are planning aids only; the final comparison must use actual full counters, actual accumulated bounds and every resumed process's cold setup. No cone selection, mass scaling or reference-guided refinement is permitted.

### Frozen full base integration
Parent pushed authorization05bbd72 for the full base H2T6 integral under the existing1e-6 outer plan and1e-8 global inner budget. Process every projected cone in its saved order, with no selection, new noisy models or enumeration expansion. Each process stops by95CPU seconds; reassess at450 cumulative integration CPU seconds. Save partial mass, bounds, costs and every failure frequently, commit after each bounded cycle, and charge every restarted engine's cold outer/inner rule setup. The author midpoint0.24815022292589073 remains only a qualified comparison within its9.04235e-6 conditional interval.

Freeze falsifiers: resume metadata must match the projected cache, order plan and source hashes, and completed indices must be a contiguous prefix without repeats or gaps. Cumulative scalar work includes every previous chunk plus current work, not just the final process. Stop dependent evaluation if independent review identifies a defect in the projection perturbation bound or10,000-operation triangle allowance. No claim of full accuracy until all42012 contributions finish.

### Full base result and limits
[measured] All42,012 cones completed across three bounded cycles of92.000,92.005 and60.709 CPU seconds, totaling244.714 integration CPU seconds. The full estimate is0.24815238374808765. No failure, unresolved cone, rank ambiguity or skipped index occurred. Source/cache/plan hashes, contiguous-prefix checks, every cycle's cold setup and cumulative operation ledger are saved in `full_results.json`. Checkpoints were independently checked for consistent counter sums and contiguous ranges; this is a bookkeeping check, not independent probability validation.

[inferred, conditional] The accumulated analytic error allowance is2.6871383094866017e-7: outer quadrature2.6865010179970986e-7, projected-ray perturbation3.1395147515197886e-12, positive inner quadrature6.058963419880414e-11 and analytic series1.6614382154418046e-24. T55 independently reviewed the projection proof and static triangle allowance in commit3d64aa2. Floating DD topology, scalar rule/bound evaluation and the analytic triangle route's numerical roundoff remain uncertified; this is not a fully rigorous enclosure of the network accuracy.

[measured] The calculation used93,181,984 analytic triangle attempts,2,732,756 positive fallbacks and48,116,528 positive quadrature nodes. Integration costs946,603,668,871 declared scalar operations; adding875,013,555 preprocessing operations gives947,478,682,426. The preprocessing allowance includes counted geometry, projection, radial planning and cold rules; integration charges every process restart again. This deliberately double-charges the planned initial cold outer rules. No opaque numerical kernel is uncharged. Operation counts use the parent ledger's arithmetic/comparison model and TRANS20, not hardware instructions. File transport, hashing, interpreter/object management and serialization are excluded from that model. Scalar resume validation and per-cone accumulation receive explicit allowances. Logged CPU includes cache loading and checkpoint I/O after module imports; it does not claim a new matched wall-clock sampling comparison. Earlier geometry/check CPU14.051, planning CPU3.708, primitive checks and pilot remain separately reported experimental costs.

[inferred, conditional] Sampling's affine arithmetic alone costs78 operations per H2T6 draw, with ReLU, argmax, target ranking and RNG free. At the full declared calculation budget, Bernoulli variance evaluated at the new estimate is1.5359371106292403e-11. Our conditional integration bound squared divided by this variance is0.00470118. This supports a conditional arithmetic comparison only, pending independent probability/roundoff review.

[measured] The estimate differs from the author's0.24815022292589073 midpoint by2.160822196922707e-6. That is inside its stated9.04235e-6 conditional interval, but exceeds our narrower bound. Thus treating the author midpoint as exact would be invalid. Using the entire author interval gives a worst squared-error/MC-variance ratio8.17163, so that existing reference does not establish an empirical MSE advantage. Neither geometry implementation is refuted by their overlapping conditional intervals.

Failure taxonomy: no computational failure occurred in this full run. The earlier strict orthant check exposed a floating representation issue, handled by an explicit projection plus perturbation allowance; it did not establish a limitation of the geometric method. Remaining alternatives are stronger independent reference integration, interval arithmetic for the existing kernel, or exact topology certification. None has been tested here and none is asserted impossible. No new enumeration or noisy sweep follows this finite validation. Root/T55 independent review and DRI classification remain separate.

Reproduce the three resumable cycles with the same command (stop once `complete` is true):
```
ulimit -v 2500000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python -m m16.approaches.small_ladder_review.portable_cost.t6.full_run
```
The ignored caches are recreated by the saved geometry/planning scripts. Resume refuses changed source/cache/plan hashes. A recorded failure requires review before restarting; already spent work remains in the saved ledger.

### Frozen omission diagnostic
Root requires resolving the2.1608e-6 difference before classification. T48 confirms that both prior reference midpoints assign zero to73,853 small positive cones, whose upper mass bound is5.003269270003126e-6. First reproduce that exact selection from its original136,438-ray cache and require exact equality of the remaining rays with its saved62,585-ray retained cache. This is reference diagnosis only; author geometry is never used to construct our DD estimate.

Two alternatives: the difference is the deliberately omitted positive mass, or the shared retained geometry/integration has another error. Discriminator: independently integrate only omitted cones using positive full5D Jacobi quadrature, with no T55 triangle or segment primitive. Begin order4, then order6 only if the summed closed q^-3 Chebyshev bound remains above2e-7 and the110CPU guard allows it. Stream at most65,536 points per batch, retain partial work/failures, and report quadrature nodes and CPU separately from the frozen production cost. Use per-cone radial bounds from the original Gram matrix, signed-orthant L1 lower bound and supporting centroid. Any material orthant violation fails; tiny floating violations remain explicit ordinary-arithmetic qualifications. The diagnostic should explain the offset within combined conditional bounds; it does not independently certify DD topology or triangle rounding. Further same-ray/interior-label checks remain separate bounded review candidates.

### Omission diagnostic result
[measured] The exact original pruning rule selects73,853 omitted cones and reproduces every saved retained ray bit for bit. Positive full-simplex Jacobi integration gives omitted mass2.4626941468808362e-6 at order4, with conditional bound5.076374514745752e-7; order6 gives2.462702926312785e-6 with bound1.0784177521623221e-7. Constant/linear simplex moment controls pass. Total diagnostic CPU63.832; node counts75,625,472 and574,280,928. This uses independent direct positive integration, not the triangle/segment kernel. SciPy rule generation and NumPy determinants are explicitly opaque diagnostic kernels; these validation costs do not alter the frozen production ledger.

[inferred, conditional] Restoring the omitted mass raises the author reference to0.24815268562881704. Its residual difference from DD is3.0188072938641675e-7, within DD plus omission bounds3.765556061648924e-7 even before the retained-reference integration allowance. The author's existing cycle15 tightens that retained integration allowance to4.03908149199579e-6 without changing its midpoint, giving combined conditional bound4.415637098160682e-6. Thus the large initial offset is explained by deliberately omitted positive mass within the combined bounds; the remaining residual is not proved zero. No new full integration, geometry expansion or noisy sweep was run. Independent source review and all previously stated floating limitations still apply.
