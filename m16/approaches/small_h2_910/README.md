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

## Cycle1 outcome
Completed in2.682CPU seconds excludingimports. All96 trained/noisy conditional
calls succeeded. Twelve independent conditional checks (393216 full inputs) had
maximum absolute z-score1.694. Gate partition error was at most2.23e-16; six
positive scale checks differed by at most2.78e-17. Results are in
`trained_tail_results.json` and use checkpoint SHA256
`004cc1af2e13b635110488da191ac3d89b32a086720225f4805200c05edb4fa1`.

| sigma | MC reference (65536 inputs) | 32-direction conditional estimate | outer SE | equal-count variance gain |
|---|---:|---:|---:|---:|
| 0 | .160248 | .144579 | .025108 | 6.67 |
| .0025 | .160248 | .144565 | .025048 | 6.70 |
| .02 | .159851 | .144743 | .022991 | 7.94 |

These coarse outer estimates are statistically compatible with the references,
but too uncertain for a precise accuracy claim. The implemented scalar geometry
costs roughly90000–98000 times one batched forward perprefix in CPU, so the
variance gain is nowhere near an equal-wall-time win. That ratio is an
implementation timing comparison, not an audited arithmetic/FLOP result.

Total primitive work including scale checks:24195 clips,1068 affine products,
1952 moment calls,1212 gatecells and740 correctpieces. There were96 outerprefixes,
6 additional scale prefixes and816 recurrent prefix steps. Reference inputs
196608 plus conditional validation inputs393216 total589824 full forwards.

Next cycle: backward target-count value-function collocation on the compact3D
state; exact terminal tail supplied byT55. Backward functions accommodate
axis/origin atoms without postulating a smooth state density. Interpolation and
terminal decision boundaries still require explicit convergence checks.

## Cycle2 protocol, frozen before transport implementation
Apply backward value-function collocation to trainedH2T10seed0 only, using T55's
forthcoming `target_probability(model,a,b,count,j)` as the terminal two-fresh-input
kernel. For each potentially predicted target j, process the earlier steps
backward on the compact3D state (h1/R,h2/R,u/R) and count0/1. Include hidden axes,
origin and target coordinate±1/0 in the grid; do not approximate a state density.
The two initial signs b=±1 each carry probability1/2. Preserve first-index ties.

Freeze nested spatial grids5 and7 nodes percoordinate;8 angular nodes per
rank-count interval, and16 nodes atgrid7 to diagnose angular error separately.
Use Gauss-Legendre quadrature in each truncated angle-CDF interval, splitting
exactly where the current input exceeds the target. Reuse terminal values for
angle refinement and report that reuse. Possible labels may be reduced only by
strict coordinatewise readout dominance; label0 is always retained fordeadstate.
The hidden box bound is B=(I-positive_part(Wh))^-1*abs(Wi) only when nonnegative
and verified invariant; otherwise stop rather than clamp an escaping state.

Two alternatives/falsifiers: (1) interpolation may smooth target/rank boundaries
so improvement with angular nodes hides a spatial bias. Compare both spatial
resolutions and angular doubling, retaining every result. (2) hidden-state atoms
may be mishandled even with a backward representation. Include all-zero readout
as a constant-class sanity problem whose exact unconditional accuracy is1/T;
check root transition probability normalization and all query coordinates against
the invariant box. No certification or solved claim follows from grid agreement.

Charge every analytic terminal call, geometry/fallback work, interpolation query,
angle node and CPU. Use no random inputs in the transport calculation; any reused
MC reference fromcycle1 is validation only. Stopcycle at120CPU seconds; if actual
terminal cost threatens that cap, savepartialresults and reduce scheduled work
explicitly, never silently reuse model-fitted values as free advice.

## T54 queued local T9 pilot protocol
Root reserved T54 for this owner; execute aftercycle2. Reuse T53's shared engine
`small_h2_78/train_pilot.py` (commit21917b4), with sequence length9, torchseed0,
training-input seed0, a biasfree torchReLU RNN with2hiddenneurons and9linear
outputs, Adam learningrate.01/defaultremainingparameters, batch256, crossentropy
against second-largest input position. Freeze max2000steps (512000sequences),
stop training at27CPU seconds with30CPU totalcap, and preserve earlystopreason.
Evaluate32768heldoutinputs with seed54003, chunks4096. Save weights andprovenance
as `LOCALPILOT_H2_T9_seed0`; this is not a published/billion-example Zoo model.
No seed selection, hyperparameter search, or accuracy-based restart.

Alternatives: weak accuracy may reflect undertraining or architectural limits;
this bounded pilot cannot distinguish them, so report trainingcurve andheldout
accuracy without claiming an expressivity limit. A numerical estimator may fail
on new geometry despite validating onT10; apply at most8fixed prefix-direction
analytic tails and2conditional32768-input checks, preservingfailures. The goal
is reproducible provenance and a first method check, not high trainingaccuracy.
