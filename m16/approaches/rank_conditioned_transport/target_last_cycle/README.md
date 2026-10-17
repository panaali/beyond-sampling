# T67: complete target-last terminal topology

## Question
Can the fixed-rank transport expose all target-value terminal breakpoints
when the correct target label is the final sequence position?

## Method
For each incoming hidden grid state at stageT-2, enumerate the two final
ReLU steps in variables (fresh scalar x, final target u). First-step scalar
gate roots partition x. Each second-step mask and target readout comparison
adds an original affine halfspace a*u+b*x+c>=0, with first-index tie ownership.
Do not reconstruct constraints from rounded polygon edges. Add x bounds
[-6,6], u bounds[-4,4], and the rank support x<=u or x>=u. When b!=0 the
halfspace gives an affine x bound; b=0 gives a u-only feasibility boundary.
All pairs of affine bound lines (including lower/upper pairs) plus u-only
boundaries partition u into slabs with fixed active lower/upper bounds.
Integrate x exactly in real arithmetic by stable Gaussian CDF differences,
divided by Z*q or Z*(1-q), where q=F_6(u). Cache these pieces per hidden
state; their endpoints augment ordinary StagePieces target breakpoints.

At u-only boundaries, active ReLU branches are strict, inactive branches
own zero, and a later readout candidate loses equality to an earlier label.
These boundary ownership flags matter for control queries exactly at u=0,
even though isolated outer-u points have zero integration mass. Internal
slab endpoints have a single owner. No arbitrary tie tolerance is allowed.

T55 independently checked exact-arithmetic completeness: active affine
bounds and feasibility change only at the retained boundaries; earlier
fixed-grid operations create no new analytic kinks beyond their own pieces.
Floating topology discovery and CDF/roundoff remain separate limitations.

## Frozen alternatives and controls
1. Missing target-last topology is the blocker: the explicit cache should
   reproduce the existing direct scalar terminal engine for all fixed
   hidden/u controls, and support complete panels without blanket rejection.
2. Positive-volume conditional ties or zero gates cause overlap: constants,
   duplicate logits, Wh0 and exact u0 queries must respect first-index ties.
3. Nearly parallel lines/roots cause numerical disagreement: preserve it,
   test exact source constraints and scale invariance, never silently merge
   roots or introduce epsilon tie ownership.
4. Complete cache cost is prohibitive: record construction/piece counts and
   query work before any full transport run. No h/q sweep. First fixture is
   public H2T4 with readout rows2/3 swapped, explicitly a synthetic control
   rather than a newly trained model. Base original also used for parity.

First cycle: primitive at fixed hidden states and u[-4,-1,-.1,0,.1,1,4],
constants/duplicates/Wh0 plus positive hidden scaling1e-16. Then build
h5/h9 topology counts only. A full outer estimate requires a separately
frozen bounded step after these controls; no new Monte Carlo or training.

## Status
Protocol frozen before implementation and measurements.

## Results
Pending; all failures and corrections retained.

## Compute accounting
Charge affine maps, gate masks, margin constraints, line intersections,
sorts, piece storage, conditional CDF queries and setup. No persistent
cross-model cache. One thread,2GB, shared locks,<=110CPU/cycle/internal100.
Event counts are not a portable total-FLOP certificate.

## Claims
[inferred] The cache can represent exact-arithmetic terminal topology.
[speculative] It makes complete target-value integration practical.

## Alternatives not ruled out
Earlier spatial interpolation, non-tensor support, adaptive outer panels,
and mixed-state forward representations remain open independent issues.

## Failure modes
Strict boundary ownership, floating parallelism, CDF cancellation, tiny
slabs, double-counted gate boundaries and exponential cell growth.

## Primitive/topology result
All fixed-u and discovered-breakpoint queries match the original direct
scalar terminal algorithm within3.61e-16; scaling1e-16 changes values by at
most1.12e-16. Constants, duplicates, Wh0 and exact-u0 first-index ties pass.
Primitive CPU.074. No tolerance-based tie repair was needed.
For the fixed row2/3 permutation, possible labels are0,1,3. At h5/h9,
complete target partitions have364/1543 panels; target-last caches contain
234/724 pieces and141/388 knots, total construction .028/.088CPU.
Numerically tiny panels occur (63/191) and must be bounded, not dropped
silently. The cache is practical at these two fixed pilot sizes.

Next bounded step, frozen before measurement: private adapter of the same
fixed_target driver, substituting only target-last terminal lookups. Compare
complete-panel h5 GL2/4, and compare cached versus original direct terminal
at the SAME GL2 nodes. Tiny panels use the prior explicit width*density
mass bound. If h5 refinement<=1e-7 and cache/direct<=1e-11, test h9 GL2/4;
otherwise retain an inconclusive result without a mesh/q sweep. This tests
fixed-interpolant functionality/convergence only, not full accuracy or a
sampling win. All cache/setup and lookup costs remain charged.

## Full h5 cache parity and retained numerical failure
Complete-panel GL2 cache/direct estimates are bit-identical at
.292265329969663, CPU1.595/2.559 seconds. Cached GL4 gives
.292265646591989, CPU3.195. Difference3.166e-7 fails the frozen1e-7 gate;
no h9 integration was run. GL2 constant target-mass defect1.060e-6 versus
GL4 8.98e-12 identifies a separate density-quadrature failure. Explicit tiny
panel omitted mass bound is1.268e-14; no mass normalization occurred.
The original readout permutation fixture has no full-accuracy reference in
this test; these are finite-interpolant results, not trained-rung claims.

One distinct bounded rescue is frozen now: map the SAME complete u-panel
endpoints to q=F_6(u), then integrate Beta(3,2) directly with per-panel
GL2/4. Its degree3 polynomial density has exact constant mass under GL2 in
real arithmetic, without weight renormalization. u stays within[-4,4], so
inverseCDF nodes avoid the previous far-tail rounded1-q issue. Reuse h5
only, same orders; this is a measure-coordinate discriminator, not a degree
sweep. Any panel collapsing to equal floating q endpoints is omitted with
an explicit original-u-width times density upper bound. Preserve original
u-rule failure. Check raw target mass and cache values; report unresolved
function quadrature separately. No h9 expansion in this rescue phase.

## q-coordinate rescue and independent handoff
Same-panel q GL2/4 estimates are .292265628860083/.292265646589676,
change1.773e-8. Raw constant-mass defects including explicit omitted slabs
are<=4.36e-14, without renormalization. qGL4 and uGL4 differ2.32e-12.
There are301 candidate panels,298 retained q panels, with3 floating
endpoint collapses; total omitted-panel bound4.008e-14. This resolves the
identified constant-density error at fixed orders, not spatial error.
No h9 integration or new accuracy reference was added.

A final adapter audit changed the last-query key from u alone to (u,q):
distinct q can round to the same inverse-CDF u but have different conditional
denominators. Initial artifacts are retained as initial_single_key_*.json.
The same full u/cache/direct and q rules were rerun after the fix: every
reported estimate is unchanged. The earlier used_panels metadata included
collapsed q panels; current metadata separates candidate/retained counts.

T55 independent review13081d0 checks original-halfspace completeness,
strict boundary ties and the q-measure rescue. It covers terminal_cache.py
and measure algebra, not the private adapter's memoization; the reviewer
subsequently explicitly agreed the (u,q) correction is necessary. Floating
root/midpoint arithmetic remains qualified. Status: bounded target-last
functionality phase complete. Original direct/cache parity, conditional
ties and measure normalization pass; finite spatial interpolation and full
accuracy remain unresolved. No broad negative or sampling win. Further
h9/target-last scaling requires a separate frozen followup.
