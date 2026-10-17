# T131: independent leaf-mass overcount diagnosis

## Question
Does the H8T4 seed0 exact-mode total-leaf mass excess come from clipping/triangulation, finite Gaussian quadrature, or both? Final3 reports1.000888 total leaf mass and attributes the defect to topology. Eight random cone Monte Carlo checks do not exclude systematic quadrature error on many trained thin cones; prior T119 already found finite mass-rule errors on H4T4.

## Method and frozen first discriminator
TASKS-only claim49947ed precedes implementation. Pin latest origin estT.py (symbolic incidence), m43 dependency, H8T4 seed0 local checkpoint and final3 evidence. Copy a read-only engine snapshot into this owned folder; never edit shared estT. No new MC, noise or full-rung evaluation.

First isolate geometry from Gaussian mass. Run exact neural gate splitting with a terminal callback that captures every terminal leaf's existing pulling simplices; replace only Gaussian orthant integration by a recording stub. For each of96 initial half-chambers, choose the fixed interior dual anchor equal to the sum of its inward facet normals. It is positive on every parent ray. Normalize every parent/leaf simplex ray to anchor dot ray=1. Sum absolute determinants of normalized rays: a correct disjoint partition must conserve this affine section volume, independently of Gaussian density. Record parent/leaf volume ratio, all node counts, and inequality violation residuals. Capture all resulting rays and their parent/leaf identities for independent same-geometry positive mass checks; no fitted selection.

Predeclared controls: orthant and one coordinate-difference cut, then record exact dyadic Fraction determinant checks for the first two parent partitions and the parent with largest floating volume residual (ties first index). Exact dyadic rays reflect captured floating geometry, not exact-real neural coefficients. Fraction volume comparison may reveal tiny arithmetic mismatches; never call approximate conservation a universal topology certificate. If a partial run reaches55CPU or200M inherited named-operation cap, save completed parents and unresolved remainder. One thread,2GB, primary compute lock.

Next scope requires freezing before implementation: replay same captured leaf simplices with independent positive Gaussian quadrature/radial bounds (T119/T123/T126), starting fixed first parents and largest geometry residual. This separates mass-rule error from geometric overcount; a numerical reference discrepancy alone cannot identify topology. Preserve source mass and every source/fallback failure. No shared repair until a concrete defect and independent review exist.

Alternative1: finite Plackett integration error accumulates on thin cells (same-geometry positive replay). Alternative2: symbolic clipping/triangulation overlaps cells (density-free section conservation, exact dyadic checks). Alternative3: terminal/output tie handling or checkpoint/source mismatch (hashes and separate gate/output ownership). These remain distinct until tested.

## Status
Protocol frozen; capture pending.

## Results
Pending bounded discriminator.

## Compute accounting
Charge checkpoint loading, engine/rule setup, all splitting/triangulation, normalization/determinants, capture/checkpoints and Fraction verification. Named-operation counters are inherited partial proxies, not a portable total. No model sampling or original accuracy benchmark is performed in this phase. Capture is new work and cannot be treated as a free reference for later methods.

## Claims
[inferred] Probability sums above one establish an inconsistency, but do not by themselves distinguish geometry from numerical integration. [inferred] Section-volume conservation is a necessary independent topology check, not sufficient to rule out compensating overlap/omission.

## Alternatives not ruled out
Improved mass rules, exact clipping, robust adjacency/rank predicates, output-tie corrections, adaptive chart selection and new representations. No whole-family negative or solved-rung claim.

## Failure modes
Floating nearly singular rays, cancellation in determinants, tolerance-based geometry, inherited incidence errors, inaccurate mass primitives and stale source artifacts. Parent anchors can be poorly conditioned; report minimum positive anchor dot product. Save partial evidence and unknowns after any cap, never silently renormalize mass.

## First discriminator result and frozen same-geometry mass test
Mass-disabled capture reproduces8455 gate leaves,43471 child simplices and all96 initial parents in13.46649451CPU. Every affine section conserves volume within1.44e-15. Exact dyadic checks give relative residuals−2.7362e-16(parent0),+3.1988e-16(parent1),−1.2156e-15(parent20,worst floating residual). Minimum normalized parent/leaf inequality is−2.72e-16; every anchor dot ray is at least.5. Geometry is floating, so these are necessary conservation/containment checks rather than a proof that no overlap and compensating omission exist. They provide no evidence of a volume overcount at the reported8.88e-4 scale.

Freeze the next mass discriminator now: use exactly parent pairs(0,1) and(20,21), the first ordered chamber and the chamber containing the predeclared worst section residual. Each pair joins the two nearest-facet halves of one full ordered-input chamber/sign chart. Their true Gaussian masses follow the binomial sign law:15/384 for(0,1),5/384 for(20,21). These are analytic controls independent of recurrent gates and Gaussian cone quadrature.

Evaluate the unchanged source Plackett formula on those captured leaf simplices, preserving every value and any negative output. Independently evaluate positive Duffy GL20 on the same rays, with exact dyadic radial bounds: if all pairwise ray dots are nonnegative, use min(norm²)/4 <= q <= max(norm²); otherwise use the finite15-support KKT minimum from the already-reviewed T34 radial primitive. Use its Chebyshev probability bound, excluding floating arithmetic. If a pair's summed bound exceeds1e-9, refine that same pair to32, then48 if needed; cap55CPU per phase and preserve partial status if still unresolved. Cache only model-independent rules. No clipping changes, truth fitting, random model inputs,1e7MC or full-model accuracy rerun. Any corrected full mass/reference needs a separately frozen phase after this causal check and independent review.

The first same-geometry mass launch failed at its first checkpoint because the sum of NumPy boolean negative-value flags was a non-JSON-serializable int64. Source/radial work for864 simplices ran but was not retained; the exception checkpoint repeated the same serialization failure. CPU for that failed process is unknown and additional to the rerun, not zero. `mass_failure.json` preserves the defect. Repair only casts the count to Python int; no numerical method changes.

## Same-geometry analytic controls: mass bias reproduced
The repaired run completes5.303671444CPU, evaluating1762 captured simplices without any geometry changes.119 require the reviewed exact KKT radial bound;1643 have exact nonnegative pairwise dots and use the acute bound.

| Parent pair | Analytic Gaussian mass | Unchanged Plackett excess | Negative source simplices | Positive GL32 error | Conditional integration bound |
|---|---:|---:|---:|---:|---:|
|0,1|5/128|+1.1257156446884e-5|99|0|3.3454e-14|
|20,21|5/384|+1.5765814580975e-5|45|−1.74e-18|1.0787e-14|

Positive GL20 already agrees to1.27e-12/8.67e-18, but both summed conditional bounds exceed the frozen1e-9 target, so both pairs were refined to32. No negative mass is clamped or discarded. Values, radial certificates, source hashes and every quadrature contribution are retained in mass_results.json.

[measured] The unchanged finite Plackett implementation overcounts two trained chamber partitions whose **same captured geometry** gives the analytic mass under independent positive integration. Therefore the final3 statement that random-cone MC checks exclude the mass formula as a source of trained-leaf bias is unsupported and contradicted for these two cases. This does not yet identify every contribution to the full1.000888 sum, or rule out compensating topology defects elsewhere. Full replay and independent review remain necessary before a global repaired reference or source fix.

Compute is13.46649451CPU capture plus5.303671444CPU successful selected replay plus the additional unmeasured failed864-simplex source/radial pass. The partial inherited source mass counter is10,334,232 named operations; independent positive setup, exact certificates and node arithmetic are separately included in CPU, not claimed free or covered by that counter.

## Frozen full same-geometry gate-mass replay (launch after independent control review)
Complete the causal test on all43471 captured gate simplices, without regenerating geometry. Eight fixed batches cover parent indices0:12,12:24,...,84:96; each has a55CPU cap and saves every completed simplex so interruption remains partial. Reuse the1762 already verified selected source/GL32 values and radial certificates, with prior costs explicitly retained. For every other simplex compute unchanged source Plackett value and an exact radial certificate as above. Choose the first order in(20,32,48) whose conditional probability bound is at most1e-9/43471; if none qualifies, retain its GL48 value/bound and mark the global1e-9 target unmet. No value clipping or renormalization.

Store every source/positive value, bound, order, index and exact-certificate method in per-batch arrays; pin input/engine/radial/checker hashes. Recompute signed sums, negative-source mass/count and total bounds using stable sums. Compare total positive gate mass against analytic1 and every two-half chamber against its exact binomial/order mass. A passing numerical reference identifies the old mass primitive's bias on this fixed topology; it is not a proof against arbitrary compensating topology defects or a cheap estimator. Checkpoint/setup and reused computations are charged, not free.

Full exact **accuracy** still requires separately captured output-error geometry, not just total gate mass. That follow-up is not silently included; freeze it after this total-mass validation and independent review. Existing10M MC is a later diagnostic with sampling uncertainty, never an estimator calibration input.
