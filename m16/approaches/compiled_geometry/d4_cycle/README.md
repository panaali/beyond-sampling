# T66 native four-input geometry

## Question
Can the checked native three-input double-description lifecycle accelerate four-input reference geometry while preserving its finite floating topology? Claim b4dcf02 was pushed alone before this protocol or code. The existing T3 and circle implementations remain unchanged.

## Method
Copy the checked lifecycle into this owned subfolder and replace its hard-coded three-dimensional storage, ranks, logit planes and constraint offsets with T=4. Keep the clipping kernel, input-gate order, depth-first traversal, incidence bits, first-index tied readout policy and pulling triangulation unchanged. Reference is the independently written Python exact_ladder/cones.py from the claimed administrative snapshot; pin its hash. No new mass formula, random model inputs, training or estimator fitting.

## Frozen alternatives and falsifiers
1. Native execution removes Python lifecycle overhead, versus clipping/triangulation arithmetic dominating at T4. Record cold and warm geometry timing, build time, serialization and rank/clip events; do not infer a portable FLOP or sampling win.
2. Apparent speed may come from changed geometry. Compare ordered headers, incidence bits, gate/outcome/simplex counts exactly, all rays/simplices within 1e-12, and hidden maps against the existing weight-derived accumulation bound generalized to three recurrence steps. Record failures before trying a separately documented repair.
3. Float sums may change near-zero decisions even with the same algebra. Before trained timing, test fixed H2 constant-output, duplicate-row, positive-rescaling and repeated-|Wi| cases, plus singular-rank thresholds. These finite controls cannot certify all topology.
4. Correct geometry may still be dominated by integration cost. Initially report geometry only. Any later end-to-end comparison must use precisely the same independently checked mass rule and charge its work; no unstable legacy Plackett value is promoted to truth.

## Frozen execution
First build and controls under the <=20CPU review lock. Then H2T4 seed0 and its pre-existing relative RMS noise .0025/.02 copies (RNG1, Wi/Wh/Wo order); compare Python/native ordered geometry. If controls pass and timing projects within the cycle limit, perform the same three cases H4T4. One cold and one warm run per implementation per case, no best-of timing. Record model/source hashes and all failures. Unsupported all-zero Wi is rejected explicitly, not counted as a successful numerical result. No H16T4 or T5 without a new bounded plan after these results.

## Status
Bounded H2/H4 geometry phase complete; independent source review pending. No mass integration or shared deployment.

## Results
Pending. Reproduction commands and raw results will be added after the first bounded cycle.

## Compute accounting
One thread; <=2GB RSS; ulimit -v2500000; numerical/build runs under codex-compute.lock (<=110CPU) or codex-review.lock (<=20CPU). Include compiler CPU and source/library hashes. Count complete geometry outputs and kernel events; inherited partial operation formulas are not a newly audited full portable count. No per-model cache or input sampling.

## Claims
[inferred] The algorithm generalizes algebraically to T4 while the two-word constraint set still has capacity. [speculative] Native lifecycle lowers cold reference cost. No numerical equivalence, accuracy, or efficiency result yet.

## Alternatives not ruled out
Different robust predicates, exact/rational topology, native higher-dimensional integration, storage redesign, lazy output serialization and T5+. Those require separate tests; finite parity does not settle them.

## Failure modes
Tolerance1e-11 classification and rank1e-9, near-identical logits, small rays, non-IEEE/LP64 LAPACK assumptions, scalar-vs-BLAS summation, gate maps with large magnitudes and serialization overhead. Neither implementation is formal exact geometry.

## Retained first reporting failure
The first control passed all geometry/map assertions but failed while computing a CPU speed ratio because the measured native CPU duration was exactly zero. The failure JSON is retained. A separate reporting repair returns null for an unresolved CPU ratio and adds the already-measured wall ratio. It does not lengthen the workload, select a faster repeat or change numerical thresholds. Build CPU2.123 seconds is recorded separately.

## Four-input parity and timing results
Six fixed analytic controls and six trained cases (H2/H4 base plus relative RMS .0025/.02 RNG1 copies) pass exact ordered headers, incidence bitsets and gate/outcome/simplex counts. Maximum ray/simplex difference is1.101e-13, below the frozen1e-12 threshold. All hidden-map differences satisfy the weight-derived accumulation screen; maximum map-bound ratio is.0265 across controls. Rank-threshold fixtures match LAPACK/NumPy, repeated input magnitudes preserve NumPy order, and unsupported all-zero input weights explicitly raise status1. These tests retain floating topology and accumulation-bound qualifications.

H4 warm geometry CPU speed ratios are26.79/32.76/23.80 on base/.0025/.02. H2 wall ratios are30.06/23.64/27.95; CPU timings are too quantized for a reliable H2 aggregate (the base native duration is zero), so no CPU speed is inferred there. All cold/warm components and serialization durations are in results_h2.json/results_h4.json. Compilation2.123CPU is additional cold setup, not amortized away; the library may be reused across models, but no model-dependent state is cached. No mass quadrature or end-to-end estimator comparison was run.

Reproduce from this folder under the stated lock and limits: `python build.py`, `python check.py`, `python check.py --hidden 2`, `python check.py --hidden 4`. Reference source is pinned by hash from admin-t66-d4-claim; checkpoint/model hashes are in each result. Build and controls used the review lock; trained runs used compute lock. The initial reporting failure remains in failure_h0_1792217246686655897.json.

[measured] Native four-input geometry preserves the frozen finite comparison corpus and materially accelerates H4 geometry. [inferred] This can shorten future reference preparation; integration may still dominate. [speculative] Larger models/T5 could benefit; not tested. No portable cost certificate, universal topology identity, accuracy improvement or sampling win follows.
