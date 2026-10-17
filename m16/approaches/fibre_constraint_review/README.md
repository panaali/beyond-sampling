# T99 approximate confirmation-order constraints

## Question
Can dropping gate-versus-gate ordering constraints on already-correct fibre
pieces reduce trace work without spoiling the passed T3 recipe? Claim4ab9117
was pushed before this implementation. Claude retains all shared ladder code
and final recipe adoption. This is a local diagnostic, not a new passed rung.

## Method and frozen first cycle
Pin fibre_fast.py, fibre.py and fiber3.py from shared commit7888dc3. Preserve
unmodified source snapshots here for reproducibility; do not edit the author's
folder. Original F2L2fast, LOOK2, is the control. The reference already makes
an attached-error/finite-lookahead approximation; it is not exact integration.

Change ONLY the forward trace's nearest-next-gate versus other-ahead-gates
constraint call. Defer this call until the piece's output intervals have been
classified. If every interval is correct, omit this one call; otherwise keep
it. Keep all entry, clamp, class-margin and reverse-walk constraints. No
changing gate dynamics, first-index argmax, LOOK, sampling or fitted thresholds.
The local implementation records dropped calls/pairs and charges its decisions.
An exact deferred-but-retained mode must agree with original before testing
omission, since operation reordering itself should preserve the recipe.

Frozen strongest cheap rescue: same omission, but when any order constraint
was omitted in a trace, cap its angular validity width at pi/64, centered on
the traced angle and intersected with remaining validity/segment bounds.
This is a predeclared accuracy/cost tradeoff, NOT an error enclosure. All three
outputs (reference, omission, angular cap) are reported; no favorable choice.

First numerical panel: public H4T3 seed0 and the FIRST sigma.02 relative-RMS
noise copy generated from that SAME seed0 by NumPy default_rng(0), each matrix
in Wi,Wh,Wo order. Never select best seed. Separate exact3.py reference on
these fixed models; charge reference separately as validation. Also zero
readout, duplicate readout, and Wh0 synthetic controls with analytic1/3
accuracy when output is a constant first-index label. Check output rescaling
1e-8/1e8 on trained seed0 as a diagnostic of inherited tolerance sensitivity,
not as license to repair source silently.

One<=100CPU primary-lock cycle,2GB, one thread. Freeze no new Monte Carlo or
training. Existing model/source setup, proxy operations, CPU, trace/constraint
counts, dropped calls/pairs, range violations and failures are retained. Broad
H4/H5/H6 seed/noise panel follows ONLY after these controls and review; all
rungs must be re-passed before DRI adoption. No blind degree or threshold sweep.

## Alternatives and discriminators
1. Bookkeeping reordering itself changes the result: deferred-but-retained
   mode compared with original is an exact implementation control.
2. Omitted constraints expose hidden wrong intervals: compare exact geometric
   truth and frozen angular cap, preserving any lost accuracy or cost gain.
3. Tolerance/scale or positive-mass ties explain differences: constant and
   scaling controls run before inference about approximation quality.
4. Gate-order constraints are not the dominant cost here: charge construction,
   remaining class constraints and trace work, not just counts removed.

## Status
Pilot and30-model discriminator completed; two independent source reviewers
and independent raw-artifact arithmetic review passed. No adopted recipe or
full benchmark pass claim. Full fixed T95 panel validation is the next phase.

## Results
Detailed pilot, scale and broader-panel results appear below. Raw estimates,
all changed cases and costs are retained in the JSON artifacts.

## Compute accounting
Source setup and checkpoint acquisition logged. Scalar arithmetic counters are
inherited proxies (multiply-add convention); special functions, allocations,
Python overhead and model-independent family construction are not a certified
instruction count. Report CPU including imports separately and add local
predicate/cap charges. Validation truth and controls remain separate costs.

## Claims
[speculative] Confirmation-piece ordering may cause unnecessary refinements
when classification is insensitive to the changed gate sequence.

## Alternatives not ruled out
Impact-aware freezing, a rigorous readout error certificate, adaptive omission
from an explicit error budget, and higher-dimensional fibre formulations.

## Failure modes
Wrong interval topology after omission, inherited fixed tolerances/caps,
false confidence from one seed, proxy undercounting, and an angular cap whose
accuracy improves only by spending more than the original recipe.

## First cycle results
[measured] All3 constant-label controls pass to1e-10 in all four modes.
Deferred-but-retained order constraints reproduce the unmodified fast recipe
exactly on both trained models and all controls. The limited omission also
preserves the two trained estimates at printed precision while lowering proxy
cost by3.81%/4.15%. This is a scoped small gain, not the suggested62% overall
reduction: only one specific ordering call was removed on correct pieces.

| Model | Numerical truth | Reference estimate | Omission estimate | Reference / omission proxy ops | Capped proxy ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4T3 seed0 | .9845503813585573 | .9845554916788959 | .9845554916788959 | 211218 / 203164 | 439885 |
| Same seed, first relative.02 copy | .8593001113320059 | .8595162512889670 | .8595162512889670 | 214688 / 205775 | 416801 |

The angular-cap rescue retains the same estimates but more than doubles
reference traces (1283/1266 versus445/475) and raises cost. It is not selected
as the default. Omission drops306/285 calls (702/497 candidate pairs), reducing
traces to434/455 and retained constraints to8746/8389. Deferred mode costs an
extra970/998 local decisions, which are charged. Output scaling by1e-8 changes
all methods by2.22e-16, but scaling by1e8 changes ALL four methods by
6.901941298265157e-6. This inherited scale-sensitivity failure is preserved;
it is not caused by the new omission. Diagnose it before broader adoption.

Whole phase3.355113876CPU including imports/setup,202376KiB RSS, no random
model inputs. Reference polygon accuracy on the original matches the pinned
ladder artifact. The source m43.py eagerly reads its baseline weight NPZ;
initial import failed because this dependency was missing from our snapshot.
No numerical model test ran in that attempt; its CPU was not measured. The
unchanged dependency was copied and hashed, while all evaluations still take
the explicit frozen models. This setup overhead is not silently treated as0.

Reproduce from the research worktree after downloading the checkpoint URL in
source_manifest.json to /workspace/work/t99_checkpoints/H4_T3_seed0.pth:

```sh
ulimit -v 2500000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/fibre_constraint_review/pilot.py
```

[inferred] The small initial gain justifies a fixed broader discriminator;
it does not establish safe omission of other ordering calls. Full H4/H5/H6
seed/noise panel, inherited scale-sensitivity diagnosis and independent review
remain required. The constant cases
are correctness controls, not representative cost measurements. Proxy MSE
ratios in pilot.json omit uncharged setup/special-function details and are
not portable leaderboard results or rung passes.

## Frozen scale-sensitivity discriminator
Before any larger panel, investigate the observed Wo×1e8 discrepancy on the
SAME original H4T3 model. Per-family comparison localizes it to the(1,2,0)
negative arc, truth2; the reference changes445 to446 traces with unchanged81
arc segments. Two competing explanations remain: incorrect geometric mass
arithmetic versus finite-lookahead decisions changing under roundoff.

Freeze controls at scales1e-8,1,1e8. First compute numerical exact polygon
accuracy for each. Run the unmodified fast engine with LOOK64, checking every
forward trace actually reaches its clamp (otherwise report incomplete and
make no exact-trace claim). Retain original LOOK2 results. This isolates
truncation decisions without changing gate-order predicates. Strong cheap
rescue: canonicalize readout by dividing Wo by its positive max absolute
entry (leave allzero unchanged), then rerun LOOK2 reference/omission/cap on
all3 scales. This preserves the real-arithmetic classifier and first-index
ties; it is not guaranteed bitwise invariant after floating rescaling.
Record normalization costs and all outputs, no choice based on truth. Same
source snapshots,<=18CPU review lock; no new models or numerical tolerance.

## Scale discriminator result
[measured] Exact polygon accuracy is stable to4e-16 across1e-8/1/1e8 output
scales. Full LOOK64 forward tracing reaches its clamp on EVERY trace (455 per
model) and reproduces that truth to4e-16 at all three scales. Its estimates
are.9845503813585574/.9845503813585572/.9845503813585574. This rules out a
large mass-integration error in this control. It does not certify every
floating predicate or arbitrary model.

The canonical-readout rescue FAILS to restore LOOK2 invariance: normalized
1e-8 yields.9845554916788958, while normalized1 and1e8 yield
.984562393620194. All reference/omission/capped variants share this split.
This finite-lookahead recipe is sensitive to small floating changes in how
its angular partition and stopping decisions are arranged. This is scoped
to the approximate recipe; real network argmax is unchanged by positive
output scaling. No generic geometric failure is inferred.

Full LOOK64 proxy costs204937/205075/205015 are even below the baseline
LOOK2 reference211218 on this particular seed: avoiding its reverse-side
confirmation work can outweigh tracing further. This is an informative
counterexample to assuming more lookahead always costs more, not a general
cost win. The bounded diagnosis used1.568498291CPU including imports; all
canonicalization arithmetic is added to proxy cost (36 operations per run).
Full panel decisions must retain the failed canonicalization and separate
finite approximation error from omitted-constraint effects.

Reproduce with scale_diagnosis.py under the same one-thread review lock.

## Independent first-cycle source review
Codex small_h2_34 independently inspected adapter.py and the proxy ratio.
The deferred call occurs after classification and before any child update or
break, with identical state/indices/functions. Thus retained-mode constraint
sets agree algebraically; floating reduction ordering is still a caveat.
Omission can affect later gates/classes outside the sampled angle despite a
currently correct piece, and the angular cap is not an error certificate.
The ratio bias^2*C/(72*p*(1-p)) is correct for a fixed deterministic estimate
versus sampling at the declared MAC proxy cost. It is not a noisy-ensemble
MSE guarantee or complete scalar-FLOP measure. This review did not rerun the
numerical panel. Source and algebra controls support the next bounded test.

## Frozen broader discriminator
After that independent review, test H=4,5,6, T=3, public seeds0..4, each original
and its first relative-RMS sigma.02 copy from default_rng(0) restarted per
model (30 fixed models). This is a common perturbation design across models,
not independent noise-replicate inference. No best-seed selection or new
training. Use the same four modes and full LOOK64 control, record that full
trace reaches every clamp, compare each to separate polygon accuracy. Report
all raw biases, proxy costs, operations per trace, and fixed-model sampling
ratios using3*H*(H+2) MACs per sample. Preserve every failed or nonfinite case.

This panel is a diagnostic step, NOT the full ladder re-pass: the original
T95 panel's additional noise draws and groups remain untested. Pinned exact
reference counters contain H4 literals in early map construction, so its
validation cost proxy is incomplete for H5/H6; it is not charged as estimator
cost or used to prove a speed advantage. Save each model before continuing;
<=100CPU total this cycle, primary lock,2GB. If time bound approaches, report
partial panel rather than select models. No parameter or scale repair chosen
based on results; failed canonicalization is not used.

## Thirty-model discriminator result
[measured] All30 deferred-but-retained outputs agree with the unmodified
recipe within1e-12. Omission lowers declared arithmetic proxy cost in every
case, by2.13% to12.29%. Six estimates change by more than1e-12; the largest
change is+4.55968358478e-5 on H4seed4's noisy copy, so this is demonstrably an
approximation. Do not report it as an exact rewrite.

| H, ten models each | Mean per-model cost saving | Reference mean proxy | Omission mean proxy | Maximum omission absolute bias | Worst omission proxy sampling ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 3.7021% | 197463.1 | 190193.3 | .00260350467 | .0981455 |
| 5 | 7.0007% | 285083.0 | 265392.4 | .000191556234 | .00193254 |
| 6 | 8.3396% | 393873.6 | 361017.8 | 2.24543556e-7 | 4.27064e-9 |

Every method has ratio<1 on all30 cases under the declared incomplete proxy.
That includes the original, omission, cap and full trace; this does not
establish improvement on the full ladder protocol. Capped mean proxies are
349549/500539/666361 for H4/5/6: the accuracy rescue is costly and does not
remove every changed estimate. All four alternatives remain in panel.json.
Full LOOK64 reaches the clamp on all16,196 forward traces and matches the
separate numerical polygon truth within3.00e-15. Its mean proxies are
200179/300751/433717, demonstrating that its cost can be close to LOOK2 in
this dimension. These facts do not extend to larger T or certify predicates.

Six omission changes (estimate minus reference):
H4seed1 noisy +5.74861018e-6; H4seed4 original +2.04333996e-5;
H4seed4 noisy +4.55968358e-5; H5seed1 original +4.73963316e-8;
H5seed2 original +8.12675705e-10; H6seed2 noisy +3.06388821e-8.
The remaining24 changes are<=1e-12. This keeps tiny introduced errors visible
rather than declaring methods equivalent from rounded aggregate values.

Panel CPU22.495051669 including imports,209288KiB RSS. Public15-checkpoint
acquisition took1.2906 wall seconds; paths, URLs and exact hashes are saved.
No model inputs were sampled. The experiment's noisy weights are fixed
benchmark instances, not estimator sampling. Initial pilot and scale control
add4.923612167 measured CPU (one failed import attempt remains unmeasured).
The source family setup is still an uncharged operation-count limitation.

Reproduce panel.py under the primary-lock command above; checkpoint URLs and
hashes are in panel_checkpoint_manifest.json. The small summary is a direct
aggregation of panel.json, retaining per-H mean savings, worst errors/ratios,
all changed cases, exact-control parity and clamp counts. Independent
numerical summary/control review requested from small_h2_34.

## Handoff and remaining adoption requirements
[measured] This finite confirmation-order omission provides a modest cost
saving on30 fixed models, with small but real added error. [inferred] It is a
candidate local optimization, not a route around the observed per-trace H
cost or proof that all gate-order constraints can be dropped. The full
original T95 noise panel, complete portable cost, inherited LOOK2 scale
sensitivity, and DRI recipe adoption remain open. No shared code was changed.
Impact-based freezing and a rigorously bounded omitted-region method remain
plausible distinct alternatives; the failed angular/canonical rescues do not
retire either family.

Independent broader arithmetic review by small_h2_34 recomputes every saved
ratio and by-H summary from raw estimates/truths/costs, all exact saved matches;
all30 deferred controls and six changed estimates verified. This independently
checks reporting, not neural integration or missing costs. T55 separately
reviews deferred placement and the scale discriminator, agreeing with the
finite-recipe interpretation while retaining inherited1e-15/1e-12 predicates
and perturbed crossing-probe caveats.

## Frozen full existing T95 panel validation
To make the candidate reviewable for DRI, run the complete EXISTING157-model
panel at each H=4,5,6,T3, in bounded50-model chunks (<=95CPU per chunk). Same
five originals, six relative-RMS noisy copies for each sigma.0025/.005/.01/.02,
and64 absolute-sigma.05 copies at RNG namespaces0 and1. The noise base follows
the ORIGINAL benchmark's fixed zoo best-seed table: H4seed0,H5seed1,H6seed3.
This is benchmark reproduction, not a new best-base selection. Preserve NumPy
noise draws per matrix/per model in original zoo.noisy_rel/noisy_abs order.

Run reference, omission and capped variant on every model; deferred equivalence
and full tracing controls are already independently checked, so no redundant
replay of those modes. Recompute exact polygon accuracy and compare the pinned
old artifact for EVERY model before using it; mismatch>1e-10 aborts that row.
Readout normalization remains off. Same source, thresholds and modes; no tuning
from this expanded panel. Save each row, count all reference and validation
work plus repeated imports/chunk setup. Fixed complete raw outputs and every
failure preserved, no truncation interpreted as a result. Report ratios per
model and per original noise group but retain incomplete cost qualifications;
no final ladder adoption without DRI review and portable cost treatment.
