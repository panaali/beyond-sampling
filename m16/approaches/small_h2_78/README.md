# T49: H2T7 and H2T8 reset decomposition

## Question
Can exact resets remove most of the history from a deductive accuracy estimate for a two-neuron RNN?

## Method
Recover the earlier exact rational readout envelope and two-negative-input reset certificate. For checkpoint H2T8 seed0, partition by the last consecutive-negative pair, analytically marginalize all discarded prefix magnitudes using T55's formulas, then integrate only the remaining suffix. First-index output ties, including a positive-mass zero state, are retained. H2T7 has no standard zoo checkpoint; acquisition/training is separately tracked in T53 and no toy checkpoint is substituted here.

Recovery cycle protocol, frozen before implementation: verify exact envelope and reset coefficients; test prefix integration at fixed positive/negative suffixes and predicted labels both before and after the reset. Use exact polynomial antiderivatives in Gaussian CDF coordinates for discarded-index targets. Compare with independent conditional Monte Carlo only as validation. Subsequently integrate suffix lengths0,1,2 with analytic or deterministic low-dimensional integration and explicitly retain all unresolved probability mass. Test base seed0 and fixed relative noises0.0025/0.02 with RNG1 if checkpoints and time permit. No full accuracy claim from a partial decomposition.

## Status
Running; original uncommitted implementation was lost in the runtime restart. The earlier broad interval [7.7497%,43.3468%] is a recorded provisional result to reproduce, not a validated claim. Root reserved the task before delegation. Claude remains DRI.

## Results
Pending recovery implementation. Standard H2T7 full-training seed0 endpoint returned404 before the outage; zoo API explicitly excludes length7. H2T8 seed0 is the fixed primary checkpoint, not a best-of-five selection.

## Compute accounting
Each experimental cycle <=120 CPU seconds, one thread, process RSS below2GB. Numerical runs acquire /workspace/work/codex-compute.lock; long runs use ulimit -v2500000. Validation chunks <=65536. Count analytic preprocessing, exact rational operations, CDF/polynomial calls, suffix model steps, and independent validation inputs separately. Rational operations are not silently equated to FLOPs. No equal-compute advantage is claimed until cost is audited.

## Claims
- [inferred] If Wi>0, the first recurrent row is nonpositive and Wh[1,1]<=0, two negative inputs reset every possible nonnegative hidden state to zero.
- [inferred] Discarded values still affect the target rank; resetting hidden state does not permit deleting them from the accuracy calculation.
- [speculative] Analytic prefix marginalization may make the short suffix contributions cheap enough to assemble an accurate estimator.

## Alternatives not ruled out
1. A compact final readout may hide substantial rank dependence. Cheapest falsifier: fixed suffix integration over all discarded magnitudes, including predictions into the discarded prefix.
2. A reset partition may double count paths or drop the no-reset remainder. Cheapest falsifier: exact sign-pattern mass accounting and constant-readout tests.
3. Most mass may require long suffixes despite frequent resets. Cheapest test: exact reset-time distribution before expensive integration.

## Failure modes
Small numerical changes can alter positive-mass zero/tie behavior. Exact algebra alone does not certify floating-point integration. The no-reset remainder is non-negligible and cannot be discarded. Deterministic quadrature remains a separately classified numerical method; the exact prefix marginalization itself does not sample model inputs. Author checks do not constitute independent confirmation.

### Recovery cycle 1 result
`check_prefix.py` reproduced the broad base/noise bounds and checked the constructive prefix integrals. There are49 rank-partition cases, 126 independent Bernoulli-convolution integrals (maximum disagreement1.12e-16), and327,680 conditional validation inputs (maximum|z|2.57). Three trained models, base and relative noise0.0025/0.02 from seed1, retain output support{0,5,6,7} and the reset theorem. Their broad interval is exactly[1111/14336,31071/71680]. All3x65,536 model-input checks obeyed the reset predictions. Input/output scale checks1e-16 and1e16 retained the support and reset-ray winner. These are author checks; the root owns independent review.

Reproduce from repository root:

```
flock /workspace/work/codex-compute.lock bash -c 'ulimit -v 2500000; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/small_h2_78/check_prefix.py'
```

Saved `prefix_checks.json` includes checkpoint SHA256 and full counts. CPU1.052s, no analytic model-input samples; independent author validation used327,680 conditional inputs and196,608 full network inputs. Rational envelope operations and comparisons are reported separately; sorting/bit complexity remains uncharged, so there is no efficiency claim.

### Cycle 2 protocol, frozen before implementation
Constructively eliminate the final suffix input: split its scalar range at ReLU roots and readout crossings, then integrate the conditional rank law in v=Phi(x). Between fixed suffix CDF thresholds and1/2 that law is a polynomial of degree at most m+2. A fixed Gauss-Legendre rule of order ceil((m+3)/2) therefore evaluates the polynomial integral exactly in exact arithmetic; this model-free polynomial rule is distinct from approximating an RNN input integral. Validate against separate adaptive1-D integration on fixed suffixes, including predicted discarded indices and ties. Integrate only suffix lengths0..3 initially and report exact unresolved sign mass. Outer quadrature error must remain separately identified.

Alternative1: the rank polynomial degree or breakpoint set is incomplete. Test explicit Bernoulli integration and deliberately mixed-sign suffixes. Alternative2: gate/logit intervals miss atoms or tiny decision regions. Test constant duplicate readouts, rescaling, and independent dense/adaptive scalar calculations before any full-model claim. Compute cap remains120CPU seconds per cycle, serial lock and memory limit unchanged.

### T53 LOCALPILOT H2T7 training protocol
Root reserved T53 before assigning this work. The missing zoo checkpoint will be replaced only by a clearly labelled local pilot, not presented as a zoo model. Freeze: H2,T7,bias-free PyTorch nn.RNN with ReLU, one scalar input per step, final nn.Linear(2,7,bias=False); torch parameter initialization seed0. Adam learning rate0.01, default other options, cross-entropy on the second-largest input position. Training inputs are iid standard normals from a separate torch Generator seed0, batch256, maximum2000 updates (512,000 sequences). Stop early only when process CPU since training setup reaches27 seconds, leaving3 seconds inside the30CPU-second cycle for evaluation/save; record actual count and reason. No checkpoint selection or hyperparameter sweep. Evaluate final model on32,768 independent standard-normal sequences from NumPy seed49003 in chunks4096. Save tiny NPZ weights, SHA256, loss trace every200 steps, software versions, actual CPU and heldout accuracy/standard error. All numerical work uses the shared flock, one thread and2GB memory cap. Apply the exact readout envelope and reset theorem test regardless of performance; if the reset theorem fails, report that rather than force this method.

Alternative1: width2 cannot learn substantial accuracy under this budget; the fixed heldout score measures this without selecting a different run. Alternative2: training hasn't converged; loss trace and actual sequence count document the limitation, but further training requires a separately recorded next cycle. This pilot's existence is useful even if its task accuracy is low; accuracy-estimation success is a separate question.

### Cycle 2 result
`scalar_tail.py` implements the exact last-input elimination; `check_scalar.py` tests six fixed suffixes, their two positive rescalings, and two constant/duplicate readout cases. Maximum disagreement against separately recomputed scalar ReLU/rank integration is2.78e-17; geometry breakpoints are shared, so this is not fully independent geometry validation. Reference checks used1,512 scalar network evaluations and105 rank-only evaluations. Final run CPU0.842s; the earlier initial check used0.903s and had unmetered validation calls, explicitly excluded from efficiency claims.

`partial.py` integrates last-reset branches with suffix lengths0..3. Their total event mass is exactly19/32; the unresolved mass13/32 includes the no-reset mass55/256. Their estimated **correct probability contributions**, summed, are0.12815997 /0.12800924 /0.12796666 for outer orders8/16/32. This is a partial contribution, NOT an accuracy estimate. The final refinement changes it by4.26e-5; that is a convergence diagnostic, not a certified error. Each of the four components agrees with separate65,536 full-input validation within1.59 standard errors. Runtime1.776s,2,744 conditional-tail calls; detailed cell/polynomial counts in `partial_results.json`.

Reproduce with the same lock/thread/memory wrapper as cycle1, replacing the script by `check_scalar.py` or `check_partial.py`. No full estimator is registered because40.625% of the input probability remains unresolved. Next mathematical cycle will exploit the exact readout envelope to avoid nonwinning-logit crossings, then extend the reset decomposition within the existing CPU cap. T53 local training is queued first as requested by root.

### Cycle 3 protocol and correction before extended runs
Review identified that equal-order tensor rules hit exact ties between known suffix inputs. Such fixed ties have probability zero under the target input distribution, but can carry nonzero quadrature weight; their conditional rank convention was unspecified. Historical `partial_results_equal_axes_historical.json` is retained. Reject tied fixed suffixes explicitly and use distinct even orders per axis (n,n+2,...) to avoid shared Gauss nodes. Recheck the affected suffix3 contribution, without treating convergence as certified error.

Precompute the true nonnegative-state readout envelope. Within each last-step gate cell, only boundaries between adjacent envelope winners can change the prediction. Compare this reduced split against all-pair splits before extending to longer suffixes. Still compute first-index argmax using all rows on each interval, preserving duplicate/tangent output ties. Preprocessing rational costs remain separate from scalar costs. Alternative explanations: omitted isolated envelope ties (test duplicate/tangent rows) or an insufficient rank breakpoint set (retain the polynomial checks). Extend only after the checks pass and T53 pilot is saved.

### T53 pilot result
The fixed run completed all2000 updates (512,000 sequences) in2.485CPU seconds; evaluation/save brought total to2.496s excluding interpreter/import startup. Final heldout accuracy is7768/32768=23.7061%, standard error0.2349 percentage points. Saved `LOCALPILOT_H2_T7_seed0.npz` SHA256 `09ac1bc640e3b15641a12606d91a4c28e699aa47fd14aff202bac52bfa0444e0`; matching JSON records the complete provenance and loss trace. This closes the missing-checkpoint pilot deliverable, not accuracy-estimation success or convergence of training. The estimator owner remains T49; root/DRI controls registry status.

Extended cycle3 protocol: after the reduced-envelope geometry check passes, evaluate all reset branches through suffix6 at distinct even per-axis orders starting2 and4. Cap each rule at45CPU seconds; discard any unfinished branch from the reported contribution and retain its entire event mass as unresolved. Record partial work in compute totals. This low-order extension is a convergence probe, not a precision claim. A separate65,536-input validation diagnoses branch contributions only. No-reset mass55/256 remains unresolved even if every reset branch finishes.

Cycle3 checks: reduced-envelope scalar geometry matched all-pair construction over102 hidden states and2,343 comparison intervals, apart from float boundary intervals totaling1.17e-15 Gaussian mass across the checks. These numerical mismatches are retained in `geometry_checks.json`; they preclude an exact-real certification claim. Five conditional probabilities agree within2.78e-17. Typical readout cells fell from23–24 to4–5. Corrected distinct-axis outer rules give partial correct mass0.12789009/0.12792549/0.12792609 at base orders8/16/32; last refinement difference6.06e-7. The affected equal-axis results remain historical. Geometry CPU0.947s; corrected partial check1.260s with2,968 tail calls and65,536 validation inputs.

The LOCALPILOT H2T7 checkpoint fails the H2T8 reset certificate, so the reset estimator explicitly rejects it. Its nonzero-state output support is{3,5,6}; a rational certificate that its hidden state cannot vanish is being added. No accuracy estimate is inferred from that support alone.
