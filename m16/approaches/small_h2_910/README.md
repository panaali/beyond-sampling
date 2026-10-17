# T50: trained H2 T9/T10 applications

## Question
Can exact low-dimensional tails and target-count transport yield an efficient
accuracy estimator for a trained two-neuron recurrent network at T=10?

## Method
This is a continuation after the runtime restart lost unpublished files. The
previous final-input code was not validated and is not used. T9 has no published
AlgZoo checkpoint (API omits T9; standard seed0 URL returned404 before restart);
root has opened T54 for that gap. No synthetic model substitutes for trained T9.

Cycle1 protocol frozen before numerical work: apply the already-reviewed T55
`feynman_fundamentals/tail.py` analytic two-input elimination to public H2T10seed0
and matrix-relative noise sigma .0025/.02 with seed1. Use32 fixed random prefix
directions (seed50110); this OUTER SAMPLING is a diagnostic, NOT a deductive
estimator. For first4 directions permodel compare against32768 independent
conditional direct executions using the exact chi8 radius law. Record all tail
failures, partition errors, gate/rank geometry counters, and CPU. Check positive
input/output scaling on2 fixed directions permodel. One separate65536-input MC
reference permodel assesses outer estimate only at its substantial sampling
uncertainty. No budget advantage inferred from equal prefix counts.

After this test, proceed to target-specific transport as coordinated with T55;
no duplicate fundamental primitive implementation. Runs acquire shared compute
flock, use one BLAS thread, at most2GBRSS and <=120CPU seconds percycle.

## Status
Running cycle1. No solved claim.

## Results
Pending. Reproduction: `python m16/approaches/small_h2_910/check_trained_tail.py`.

## Compute accounting
Count random outer prefixes, deterministic recurrent steps perprefix, full model
validation/reference inputs, polygon clips, affine products, weighted moments,
and CPU separately. Geometry operations have not been converted to auditedFLOPs.
This study makes no equal-compute win claim.

## Claims
- [inferred] Conditional tail elimination is analytic; random outer prefix
  directions keep the full estimator sampling-based.
- [inferred] Target-specific accuracy needs hidden state plus a capped exceedance
  count, with one coordinate for conditioned target input; H2 gives3 continuous
  coordinates regardlessT. Low state dimension does not imply smooth density.
- [speculative] Reusing singular transport strata can amortize per-prefix geometry.

## Alternatives not ruled out
1. A primitive valid on synthetic cases may fail trained thin regions or reset
   ties. Cheapest check: trained/noisy conditional comparison, partition mass,
   explicit failure preservation and scaling check before any full estimate.
2. Marginal variance reduction may be overwhelmed by geometry cost. Cheapest
   check: separate measured CPU and all geometry counters; do not price an
   analytic tail as one forward evaluation.
3. Conditional independence may be applied to an absolute prefix incorrectly.
   Validation redraws its chi8 radius rather than holding raw prefix fixed.

## Failure modes
AbsentT9 checkpoint; floating clipping tolerances and cancellation; trained
singular/axis atoms; substantial uncertainty from32 outer directions; scalar
Python geometry overhead; no certified bound on floating integration error.
