# T100: recovery of the frozen H8T4 fibre evidence

## Question
What has actually been completed for the fibre method at H8T4, and which
accuracy, convergence, noise and cost checks remain before a rung judgment?

## Method
TASKS-only claim53b895f1 precedes implementation. Read original
understand_ladder files without editing or rerunning completed experiments.
Pin saved results, logs, source snapshots/history and all other local result
variants. Recover the intended recipe (LOOK3, adaptive spherical cubature),
MC truth/cost conventions, and any missing noise rows. Contract the existing
five seed results independently. Compare saved seed0/seed1 values to the
already independently reviewed positive exact references from T131/T148,
with their conditional integration bounds and provenance limitations.
Do not treat MC variance subtraction with a zero floor as proof of zero
error. Sampled gate-pattern counts are lower bounds on region counts.

Before any new numerical discriminator, freeze its scope based on recovered
evidence. No duplicate long seed run, new MC, Gaussian mass replay, shared
source change, T108 certificate work, or DRI estT_fix re-score is authorized
by this initial recovery phase.

## Status
Saved-data recovery and analytic zero-normal repair are independently reviewed
(2827c813). Observer/preflight review passed88490d79. The first missing noisy
case is running as48 resumable face-side components; no noisy-rung judgment.

## Results
Five clean rows recovered without reruns; historical group ratio.01796 is
qualified by legacy costs and MC truth. An allzero clipping bug was found and
locally repaired; three nonzero analytic controls preserve outputs bitwise.
The fixed first noisy component completes7.13CPU with no observed cap.
Detailed protocols, raw records and limitations are below.

## Compute accounting
Inventory, analytic controls and the frozen noisy component runs are logged
separately below. One thread,
2 GB, <=20CPU review cycles. Native kernel/cubature/trace costs stay qualified;
the historical hand counter and its sampler convention are retained exactly.

## Claims
[inferred] Existing saved work can distinguish a missing experiment from an
outdated narrative without repeating expensive evaluations.
[speculative] Corrected independent references may materially change how the
existing fibre outputs compare with sampling at their actual cost.

## Alternatives not ruled out
Source/configuration drift, different checkpoint bytes, finite cubature bias,
lookahead truncation, walk/subdivision caps, missing noisy outputs, and a
large gap between the historical and full operation counts. Snapshot parity
is not by itself complete execution provenance.

## Failure modes
Promoting a low-precision printed log to exact output; inheriting biased
references; confusing a sampled pattern lower bound with a complete region
count; assuming the T3 fast implementation is the T4 implementation; treating
point agreement on two seeds as a full noise-panel or convergence result.

### Recovery result
37 local copies have one identical saved H8T4 result file: five clean seeds,
no noise rows. The noisy log is empty; the clean log contains only seed0.
The five source/helper files exactly match the latest result commit5906384b
(snapshot consistency, not full execution provenance). The original rows
contain no checkpoint/source hashes. Historical estimator wall time totals
4257.27 seconds; no run was repeated.

All saved MC standard errors reconstruct n=1e8. Historical variance-subtracted
ratios are .04045/.11989/0/0/0 with group .01796 under the original320-unit
sampler and hand counter. Zero floors are not evidence of zero bias. Against
reviewed positive references, saved seed0 error is+2.417276456e-6 and seed1
+6.711647826e-8; conditional reference integration uncertainty is retained.
The legacy cost ratios are informative historical arithmetic only: actual
source cost, noise robustness and original checkpoint provenance are not
certified. A complete rung claim is unwarranted.

A structural reporting error is identified: sampled gate patterns are a
lower bound on true regions, so face_cells/patterns is an UPPER bound on
face_cells/regions. Values1.40..1.67 cannot prove that boundary face cells
outnumber all linear regions. The original summary header reverses this
inequality. This limits the T4 scaling inference, not the measured counts.

Reproduce inventory with `python inventory.py`; exact stored-float contraction
with `python contract.py`. Read-only CPU .024933/.001041 seconds.
Independent review requested before conclusions are registered.

### Frozen analytic continuation (TASKS77504a05)
Four H2T4 Wh=0 controls: all-zero readout, unique constant class0, last-input
sign classes(+3,-0) and(+0,-3). Their exact Gaussian accuracies are1/4,1/4,
5/16 and3/16. For the last input being second, positive sign probability is
int[1/2,1]3(1-u)u²du=11/64 and negative5/64. Symmetry among the other three
indices gives7/64 positive and9/64 negative per index. This uses Gaussian
CDF rank transformation, not numerical quadrature. Readout ties in the
all-zero model select index0 exactly.
Both unchanged F2L3 andFX are tested; freeze1e-7 absolute diagnostic tolerance
before outputs, record all discrepancies/costs and preserve partial arms.
Each case is separately bounded<=110CPU, one thread2GB. No full trained
run or sampled input is performed. Compare tiers to distinguish lookahead
from common geometry/cubature error, without declaring a method dead.

### Analytic result and scoped defect
The unchanged all-zero network returns1.0 in both tiers, instead of1/4,
with zero traces/pieces. `fibre4.clip` normalizes an identically zero normal,
producing NaNs; all vertex-side tests fail and both polygon outputs vanish.
`cells_on_face` therefore discards all geometry. This is a concrete zero-row
implementation defect, not evidence against the fibre method.

The other three controls pass both tiers, which agree bitwise: constantclass0
error2.67953e-10, (+3,-0)1.94819e-10, (+0,-3)2.75244e-10. These validate finite
normalization/sign/rank controls only; thin trained cubature and lookahead
remain unverified. CPU allfour cases8.35 seconds approximately; raw per-arm
counts/timings are preserved in analytic_*.json. No sampled inputs were used.

Plausible local rescue: treat an exactly zero halfspace normal as containing
the entire polygon on one side, with no duplicate complementary polygon.
Check exact component zeros, not a small-norm tolerance; nonzero underflow
would require separate stable normalization and remains open. Preserve
original failures, apply the guard only in this owned review process, and
recheck the same four analytic cases before proposing a shared-source fix.

### Local exact-zero rescue
The owned-process guard returns each exactly zero halfspace as(poly,[]),
keeping its full geometry once. Zero-network accuracy becomes.25000000996971117
in both tiers (error9.97e-9, within frozen1e-7 tolerance), with144 traces/108
pieces. Across both tier calls1536 zero normals are handled. The three
nonzero controls retain bitwise identical numerical answers and no guard
activations; a conservative legacy6-unit guard charge per3-vector is included.
Guard controls total9.09CPU approximately. Original source remains untouched.
Independent review and DRI adoption are tracked under T169; nonzero norm
underflow, trained/noise behavior and complete cost remain open.

Reproduce source controls with `python analytic_controls.py CASE` and local
rescue with `python guard_controls.py CASE`, CASE=zero,constant0,plus3_minus0,
plus0_minus3; use one-thread resource/lock limits from the protocol.

### First missing noisy-case cost preflight (board e77a8098)
The independent T100 analytic/source review is2827c813. The original missing
case `abs0.05_n0_0` is now pinned in `noise_identity.json`: local seed1 checkpoint
34fbb32d51cfae0d77b6cba73747506a93a72077657c4f02e243074178b75761,
float64 Wi/Wh/Wo model bytes ef22df10439aa5f65366a09e3cf1a42b3a8874bb1f7ee5eea68823bfaf52c9e9.
Noise is original default_rng(0), sigma.05, first copy, after float64 conversion;
independent explicit generation agrees exactly. No model inputs were drawn.
The best-of-five mapping selects seed1. T131/T145/T148/T109 references are
clean, final3 noisy seed0 sigma.005/RNG7 differs, and T113 is T3. A search of
both current source and910 approach JSON/README files finds no matching model
hash; generic matching noise labels belong to T3, not this checkpoint.

`observer.py` adds counters only to existing trace/walk64 loops and subdivision
depth6 termination; numerical statements, branches, thresholds and return
values remain unchanged. For-else counts actual loop exhaustion, not a final
iteration that exits normally. The exact-zero guard is identical to the
reviewed local rescue. `observer_controls.py` replays all eight saved analytic
arms and checks answers, legacy counters and trace/piece stats bitwise.
All pass in9.82CPU, maxRSS36MB; max trace iterations2, max subdivision depth3,
no exhausted cap. These controls do not exercise the downward walk, whose
source transformation still needs independent review. Observer bookkeeping
is review overhead and is not added to the candidate legacy cost.

After independent source review, `noise_preflight.py` evaluates ONLY family0,
side0 F2L3. It is one additive error contribution, not a complete accuracy.
One thread2GB/110CPU maximum, frozen LOOK3/ADAPT/default orders/EMAX. No
fresh MC, complete noisy sweep or clean rerun is authorized by this phase.
A killed attempt retains its pre-run artifact; normal completion records
legacy cost, counters, timing and cap observations. Alternative explanations
for poor noise performance include (a) face/trace explosion, checked cheaply
by this fixed first component, and (b) unchanged cubature/lookahead bias,
which this preflight cannot establish or exclude. A reviewed bounded full
case and independent truth/convergence discriminator remain necessary.
Reproduce identity with `python noise_identity.py`, observer parity with
`python observer_controls.py`, and frozen component with
`python noise_preflight.py` under the stated locks/resource limits.

The source gate passed before the fixed noisy component. Family0/side0
completed in7.129CPU,36.8MB: raw error component.18712027456605582,
normalised.009479623851256004,37,704,799 legacy units,2631 traces/2163pieces.
No zero normal occurred. Trace max11 (2631calls), downward walk max1
(784calls), subdivision maxdepth3; no observed loop/depth cap. This makes
one whole fixed noisy case a plausible bounded continuation; no inference
about the other47 component costs or approximation error is made yet.

Board86f0a9f1 freezes the remaining47 components, preserving the preflight
component0 exactly once. `noise_components.py INDEX` uses family=INDEX//2,
side=INDEX%2, INDEX1..47. It calls the same F3.face_err with identical initial
Ops/stats and pinned observer; no source settings change. Each process runs
with one thread/2GB/110CPU, writes an attempt before compute, then complete
raw fields or exception evidence. Resume never overwrites an attempt.
Final contraction must use original ascending family/side scalar addition,
not sorted-by-value or a selected partial sum. Full noisy accuracy remains
unavailable until all48 components and independent aggregate review finish.
