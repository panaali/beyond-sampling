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
Completed bounded reference repair; full gate aggregate independently audited f0f4012, output aggregate independently audited8f90409.

## Results
See measured gate and output-error results below; exact commands: capture.py, mass_check.py, full_mass.py BATCH (0..7), summarize_full.py, error_capture.py, error_mass.py BATCH (0..7), summarize_error.py.

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

A concrete single-simplex failure is preserved in `negative_mass_fixture.json`: captured index672 has unchanged source probability−4.8037712502280e-7, while positive GL32 gives1.4730961003631e-11 with conditional bound1.87e-23. Selection was posthoc for bug reproduction (most negative of the1762 frozen controls), not estimator tuning. A negative Gaussian mass is already invalid independent of neural topology.

T55 independently recomputed all1762 exact captured-ray determinants (separate24-term Fraction expansion), all1643 acute and119 KKT certificates,60-digit tail bounds and saved sums in14.9CPU. The analytic controls and scoped attribution pass; full replay may proceed. Review commit follows. Floating arithmetic and geometry limits remain explicit.

Full replay batch0 (parents0:12) completes7303 simplices in15.723050201CPU. Old source sum .23451366176179222 versus positive .23437499999999978 (analytic15/64), conditional bound1.2133e-11, no per-simplex bound misses. There are401 negative old values, summing−4.14196e-5; none is clamped.864 prior controlled values are reused with earlier costs retained. Remaining84parents are not included in this partial total.

## Full same-geometry gate total
All eight frozen batches complete: every one of43471 captured simplices appears exactly once. The unchanged source sums to **1.0008880231700568**, reproducing the reported overcount. Independent positive integration on the **identical rays** sums to **.9999999999999994**, with summed conditional integration bound8.5571005563e-11. All48 paired ordering/sign chambers agree with their analytic binomial masses within4.38e-16. There are2368 negative old-source values, signed sum−.0002575353588722262; all are retained. No bound misses or negative positive-rule results. Orders20/32/48 occur19916/23552/3 times;1762 controlled values are reused with earlier costs retained.

[measured] The numerical mass implementation accounts for the entire reported gate probability excess on this fixed captured geometry. The assertion that cone mass is excluded as its cause is refuted. This does not prove topology exact in general or exclude compensating defects, and gate mass is not classification accuracy. Source shared files remain unchanged. This positive reference is not a cheap competitive estimator.

Full replay costs104.109653206CPU, in addition to13.46649451CPU capture,5.303671444CPU selected successful replay, .37828463CPU aggregation, and unknown CPU from the preserved failed864-simplex pass. The full old-source partial counter246709164 does not account for positive quadrature arithmetic; no portable-cost claim uses it. T55 independent saved-array aggregate review requested.

## Frozen output-error follow-up
After the independent full-total aggregate review, capture the pinned exact engine's actual output-error simplices for all96 initial halves. Preserve its `_cell_error` callback and replace only `orthant` by a recording stub; wrap the terminal callback to log identical readout-row/tie cases without changing classification. Count gate leaves, output simplices, zero-hidden cases and exact-equal readout pairs. First-index all-off behavior is already explicit in the pinned engine. Any other identical earlier readout producing positive-volume tie is an unresolved semantic concern, not silently corrected.

First capture is capped55CPU/200M inherited geometry operations, checkpointed by complete initial half; failures preserve partial coverage. Then replay unchanged source mass and independently positive quadrature, using the first order20,32,48 whose conditional per-simplex bound is at most1e-9 divided by the captured error-simplex count. Work in fixed12-parent batches, each55CPU. Preserve every negative, singularity, exception, unmet bound and partial result; do not clamp or normalize. Sum error probabilities and report1−error only after all parents complete. Pin all inputs and charge the new capture, certificates, rules and setup. Existing10M torch and4M numpy MC serve only as uncertainty-aware external diagnostics, never fit/tuning targets. No reuse of gate-total values is presumed because output clipping changes simplices. The experiment validates a reference and diagnoses a defect; it does not establish estimator competitiveness or authorize unreviewed shared-source edits.

## Corrected H8,T4 seed0 reference
The pinned original output-error geometry completes all96 parents,8455 gate leaves and35098 error simplices in14.991144678CPU. No all-off gate cells or coefficient-identical active readout rows occur; explicit all-off first-index controls pass. The error callback itself is unchanged. Independent T55 source review confirms capture/callback and sequential competitor-union semantics; T139 separately checks fixed interior witnesses against direct network evaluation.

All eight replay batches complete in81.39114827CPU. The unchanged source yields error .024952664060133686 and accuracy **.9750473359398664**, exactly reproducing final3. Independent positive integration on the same error simplices yields error **.02383919245339731**, hence accuracy **.9761608075466027**. Its summed conditional integration bound is1.04221660744e-10, with no bound misses. Orders20/32 are used26037/9061 times. Source2984 negative terms sum−.0002925205562385668; all remain in the signed source total, none is clamped. The numerical source overstates error by **.0011134716067363742**.

The corrected reference differs from pre-existing4M numpy sampling .97606325 by+1.28 plug-in standard errors, and from README's rounded10M torch .97617 by−.19 standard errors (that printed number has rounding halfwidth .000005). These are external diagnostics with sampling uncertainty; no fitting, new sampling or threshold selection uses them. T55 independent saved-array aggregate review is pending. `error_summary.json` retains raw totals, costs, provenance and uncertainty calculations.

[measured] Replacing only Gaussian mass integration repairs both the gate total and this model's accuracy discrepancy. [inferred] These results strongly support a finite mass-rule defect as the cause of the published computational discrepancy; independent high-precision T137 review distinguishes quadrature truncation from covariance arithmetic on a concrete negative fixture. Floating geometry/roundoff certification and arbitrary other models remain outside this result. Positive integration here is a reference repair with substantial cost, not evidence of a challenge-winning estimator. The captured source named mass counter204957048 omits positive quadrature cost and cannot be reused to claim competitiveness.

Independent T55 output-array review8f90409 verifies all35098 indices and exact sums of stored binary64 values, bounds,2984 negatives and source reproduction. T139 separately verifies all35098 fixed interior direct-network witnesses and64 small-margin exact Fraction witnesses (point-panel scope only). T131 is complete within the stated conditional reference scope; board closured9240ad requests Claude/root registry acceptance.
