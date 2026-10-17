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
Inventory and independent saved-data contraction in progress.

## Results
Pending. The README's one-model prototype statement is stale relative to the
saved five clean H8T4 rows; noisy work must be checked from actual artifacts.

## Compute accounting
Read-only inventory and saved-data arithmetic only in this phase. One thread,
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
