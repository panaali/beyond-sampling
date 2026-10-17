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

## Cycle2 outcome
`backward.py` and `backward_results.json` record the first deterministic backward
transport application. The exact terminal kernel is T55 commit23b2291; fallback
dependency86c3759. The source hash is saved with the result. Run with one thread
and the shared compute lock; `FUNDAMENTALS_DIR` can select that dependency folder
when using an older isolated worktree.

| hidden/target grid | angle nodes per rank interval | estimated accuracy |
|---|---:|---:|
| 5³ | 8 | .142955257 |
| 7³ | 8 | .150793513 |
| 7³ | 16 | .150857393 |

The independent cycle1 reference is .160247803 with SE .001432963. Spatial
refinement changes the result by .00784, versus .000064 from angular doubling;
this is not convergence at the desired precision. The constant-class control
returns .096628695 and .098598010 instead of exact .1, directly exposing rank
mass distortion from interpolation. No neural prediction error is involved in
that control. The final trained estimate remains about .00939 below reference.

All3768 terminal calls (3744 trained-model grid points/counts plus24 constant
controls) succeeded. Maximum gate partition error was4.45e-16; no positive
quadrature fallback occurred. Work, counting the grid7 kernel only once despite
angular reuse:572604 polygon clips,57736 affine products,282626 readout-margin
products,74690 weighted moments,621824 angular nodes and624753 interpolation
queries. Total CPU was76.746 seconds including checkpoint loading; no random
inputs or full-sequence forward evaluations occurred in this cycle. Terminal
caches are saved solely for explicitly charged reuse, not free model advice.

[inferred] The low-dimensional representation is operational on a trained T10
network, but this coarse interpolation does not yet give competitive accuracy.
[inferred] Preserving rank-event mass is a concrete next improvement; it does not
by itself address hidden-state interpolation. Next numerical task is the queued
T54 local T9 pilot, followed by a separately frozen rank-conditioned variant.

## Cycle3 protocol: rank-conditioned backward values
Frozen before this variant's implementation/run. Reuse both recorded terminal
grids, charging their original model-specific geometry in any complete-estimator
cost. Let q_m^k(b,K) be the model-free probability that exactly one of all
non-target inputs exceeds the target after m remaining fresh inputs. Propagate
W=V/q instead of V. At each random step weight each quadrature node by the next
q and normalize these positive weights to sum one. Deterministic target
insertion leaves q unchanged. Use the exact initial masses
q(+1)=2/T*(1-(T+1)/2^T) and q(-1)=2*(T+1)/(T*2^T), each sign weighted1/2.

Compute q directly at every distinct b/angle query by positive generalized
Gauss-Laguerre quadrature (R²/2 has Gamma(k/2) law), with32 and64 nodes as an
explicit sensitivity test. Queries depend only on b, k, remaining count and
angle, so share them across hidden-grid coordinates and target passes. No q
interpolation is used. Keep the previous linear spatial interpolator and16nodes
per rank interval. Test both spatial grids5/7 and both rank rules32/64.

Falsifiers: (1) constant W=1 must remain1 to roundoff; otherwise the conditional
weights or capped counts are wrong. (2) Gaussian-rank quadrature may introduce
its own bias: report rule sensitivity, terminal W excursions outside[0,1], and
normalization defects before row normalization. (3) apparent improvement may
leave hidden spatial bias: preserve both grid resolutions and all original
uncorrected results. No solved claim or free-cache efficiency claim. Initial
cycle cap120CPU seconds under the same lock/thread/memory limits.

## T54 pilot outcome
The fixed run completed all2000steps/512000trainingsequences in2.507CPU seconds;
training plus heldout evaluation/save took2.520CPU seconds. Heldout accuracy was
6020/32768=.18371582 (SE .00213929). This is a weak, locally trained pilot;
undertraining and architectural limits remain unresolved. Full provenance,
versions and loss trace are in `LOCALPILOT_H2_T9_seed0.json`, with small weights
in the matchingnpz; SHA256
`38a8174ef25ace94e84ad18ffa7f1b3c038b20da652fe3b51646a7a0330cab3c`.
Reproduce with `python m16/approaches/small_h2_910/pilot_t9.py` after integrating
T53's `train_pilot.py`. Training implementation hash is recorded in provenance.

All8 analytic conditional-tail calls passed; their2 independent32768-input checks
had z-scores−.674 and1.178. Partition mass error was at most4.45e-16. The8-direction
outer diagnostic was .167269±.032289SE, too uncertain for a precise estimate.
The follow-up took .159CPU seconds and1672clips,80affine products,145moment calls,
plus56prefix steps and65536full validation inputs. These are additional to the
training examples/gradients and32768heldout inputs. T54's checkpoint gap now has
a reproducible LOCALPILOT artifact; the accuracy-estimation problem remains open.

## Cycle3 outcome: rank normalization does not fix neural interpolation
`doob.py` / `doob_results.json` preserve all four frozen variants. At32/64rank
nodes, grid5 accuracy was .137260857/.137261377 and grid7 was
.148063152/.148063575. These are worse than the unconditioned estimates despite
fixing the constant-class control to .1 within3.49e-9 (constant W within2.78e-17).
Changing the rank rule moves the final result by at most5.21e-7, so this tested
rank-quadrature resolution does not explain the remaining percentage-point bias.

The normalized terminal W exceeded1 by up to .000197 at32nodes and .00000934 at64;
no values were clipped. Before row normalization, model-free rank mass differed
by at most .000450/.000236 in absolute terms, with large relative defects on
rare states. These diagnostics preclude calling the numerical weights certified.

The incremental cycle used1.636CPU seconds, zero random inputs,267264rank
quadrature nodes,1536rule-generation nodes,1257984angular nodes and1262712
interpolation queries. It also recomputed24constant-control terminal calls
(1440clips,192affine products,324moments). The previously charged trained terminal
caches remain part of any full-estimator cost. [inferred] Hidden-state/value
interpolation, not just lost rank mass, remains a material obstacle.
