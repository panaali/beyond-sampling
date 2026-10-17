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
