# Frozen reconstruction experiment

Claim4226647e follows pilot handoff603fe260/39a19090 and independent75da4bad.
Recover_h4 owns the source/resource gate; recover_910 owns saved verification.
No execution before both the source clearance and immutable fullfit_gate.json
are published. This is one fixed reconstruction experiment, not an estimator
submission or recovery of the missing historical fit driver.

## Data and fits

Use the existing original checkpoint's exact binary weights from the validated
pilot weights archive; no reload, new model, noise or reference forward.
Use precisely65536 train inputs and two independent65536 holdouts with the
already declared PCG64 seeds2026101802011/12/13 and NumPy2.5.4. Reuse the
original1024-input pilot prefix and its candidate hidden/logit arrays once.
Resume its stored RNG state, then generate63 remaining training chunks and
64 chunks for each holdout:191 new chunks,195584 new vectors,1955840 normal
values. Each chunk is1024x10. Preserve before/after generator states and full
input/hidden/logit arrays. Stop on any input tie without resampling/exclusion.

Fit the unchanged reviewed core helpers with rcond1e-12:216 homogeneous
three-feature state-specific maps from(m1,m2,x9) to16 hidden coordinates,
and one18-feature case2-only map from m2*onehot(p2),gap*onehot(p1). No intercept,
ridge, extra ReLU, merged states, feature selection or tuning. Save both complete
coefficient arrays, all216 counts/ranks/singular values and case2 fit metadata.
Underfilled or deficient designs stop dependent work. Coefficient archives are
exclusive and their hashes must remain unchanged for every scored panel.

## Scores and uncertainty

Score train and each holdout separately. Lookup hidden MSE/R2 is over the full
panel; separable hidden MSE/R2 and classification scores use case2 only. Use
the actual test mean for each R2 denominator; retain negative values, represent
zero-variance R2 asNone and exclude only thoseNone values from its median.
Report network agreement and task accuracy separately, with exact counts and
denominators. Task accuracy alone cannot establish reconstruction because the
state already contains the true index. Read raw fitted hidden vectors through
the original readout without rectification. Preserve first-index readout ties.

Save complete original-network and lookup K-bin(0,1,>=2) x case x predicted
GLOBAL rank tables, plus the restricted separable table. Retain per-observation
true/network/lookup/separable labels, cases, sign bins and state keys. Derive
an explicitly disjoint network error partition: blind(K<=1) errors; nonblind
case0/1 global-third errors; all remaining nonblind errors. These sum to all
errors. Retain the full table rather than reconstructing the old rounded.96%
prose bucket. Report K0/K1 conditional error counts separately.

On holdouts only, use ordinary numerical equal-tail95% Clopper-Pearson intervals
for each task-accuracy/agreement and blind-error proportion. These condition on
the fixed trained coefficients/model and each stated denominator. They are
individual intervals, not simultaneous coverage, fit-variability intervals or
an outward numerical certificate. No training fitted-score confidence interval
is reported because the fit depends on the same observations. Four frozen
boundary controls precede model work: n0; k0/n1; k1/n1; symmetric k5/n10.
Preserve all failures; no confidence-level/sample-size adaptation.

## Evidence and alternatives

All raw inputs/hidden/logits, fitted coefficients and per-observation labels
are retained. A reviewer can reconstruct predictions and scores without
refitting or new RNN calls. Unlike the synthetic controls, fitted coefficients
and their evaluation inputs are available. Derived fitted hidden/logit arrays
need not be duplicated because the complete features/coefficients are saved.

Alternative explanations include in-sample optimism (separate fixed holdouts),
network/task conflation (separate agreements and accuracies), overlapping error
buckets (full disjoint tables), accidental coefficient or stream changes
(immutable hashes/state chains), and numerical wiring errors (validated pilot).
Historical provenance and untested reconstruction families remain open.
No choice is made using measured performance.

## Resources and failure accounting

One process with one thread, primary lock,110 CPU-second TOTAL reserve and
external limit;90-second soft total guard. No multiplying the reserve by chunks.
2500000KiB virtual-memory shell cap;2GiB RSS completion check, as in the pilot.
Use Bash timing. Imports, input generation, forwards,217 actual lstsq calls,
prediction/scoring, interval calls, compression and serialization count toward
research CPU. Core call counts retain actual partial lstsq/forward progress;
two fit-helper batch attempts are separately recorded. Inputs are saved before
each forward, arrays before scoring, and progress/failure records are durable.
No automatic restart or partial replay; any interrupted continuation needs a
separate recovery scope preserving successful work.
Reject any preexisting content in the output directory before creating progress
or generating inputs. The internal CPU timer begins inside main and therefore
excludes interpreter/top-level standard-library imports and final result,
progress and stdout writes. External Bash timing and the110CPU process limit
include that overhead; no claim is made that the internal timer is total CPU.

Total candidate inputs including the reused pilot are196608; this new process
forwards195584 only. Prior library reference1024 remains separate; this process
has zero library/reference forwards. The5948-unit canonical forward benchmark
alone gives1169424384 units for the full candidate data, not a complete cost
or sampling-comparison claim. Independent saved audit precedes conclusions;
neither success nor failure establishes a general impossibility or ARC rung.
