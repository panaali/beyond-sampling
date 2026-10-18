# T189 exact whole-box mass discriminator

## Question
Does the fixed zero-anchor/global-envelope conditional Chernoff bound leave a
provably positive, practically relevant amount of T4 input mass undecided?

## Method
Claim2821fc26 precedes code. PROOF.md derives the sufficient all-rival criterion
and disjoint top-two box mass. box.py uses only exact saved global bounds and
rational Gaussian CDF enclosures. run.py freezes four predicate/five CDF controls,
then every72 pair check on H4/H8 T4 seed0, with no network calls.

## Status
Frozen run and decode-only recovery completed; independent saved-result review pending. Source gates2af8ef71/39230666 and recovery pin065c4302 are published.

## Results
[measured] All72 exact pair inequalities completed. H4T4seed0 certifies4 disjoint boxes, probability at least0.08526780093644182 (display); H8T4seed0 certifies2, at least0.04263390046822091. Exact outward rational endpoints are in recovery_results.json. Remaining boxes are unresolved by this sufficient predicate.

The maximal Bernoulli sampling standard error is smaller than these width lower bounds at N>=35 and N>=138, respectively. This is an interval-width diagnostic, NOT a matched-compute/MSE comparison. It does not prove actual estimator error. All models are seed0/T4; no scaling/all-seed claim.

Executed once: `ulimit -v 2500000; ulimit -t 20; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/conditional_order_bounds/box_mass/recover.py` (after failed original run.py). Existing output guards prohibit repeated execution. All9 completed controls and mass were reused exactly; no successful calculation was repeated.

## Compute accounting
Reused T187 global-envelope setup, new exact predicate operations, rational CDF
construction and MP validation are recorded separately. No native-equivalent
price or matched-budget comparison is claimed. Named comparisons cover the two sufficient-test conditions; input guards, validation comparisons, indexing and conversion remain unpriced.

## Claims
[inferred] Passing all rival inequalities certifies whole-box vacuity of this
specific Chernoff family; interval integration has at least that width.
[speculative] The frozen boxes may expose a decisive limitation cheaply.

## Alternatives not ruled out
Rank/domain/reachability restrictions, changed anchors, one-sided allowances,
adaptive retained coordinates and methods other than this global envelope.

## Failure modes
Incorrect age/index mapping or pair lookup, adding omitted-rank conditioning,
forgetting zero-allowance ties, double-counting boxes, or interpreting uncertainty
width as actual estimator error. Source/control gates and independent saved audit
are required. Scientific positive/negative claims stay scoped to fixed models.

## Preserved first attempt, 2026-10-18
Source gate2af8ef71 and protocol2708cc95 preceded the run. Four predicate and five CDF controls passed; the first saved-model rational decoding failed because Fraction requires integer components while T187 saves decimal strings. All controls, mass, failure, progress and 0.139998291 process CPU seconds are preserved in results.json/progress.jsonl/run1.log. No model inequalities were evaluated. recover.py freezes integer decoding only, retaining original source and reusing all completed controls/mass; independent recovery gate pending.

## Recovery accounting
Recovery processCPU0.167064820; first attempt0.139998291; combined about0.307063111 including separate startup imports. No neural forwards, checkpoint loads, random inputs or reference quadrature. Original and recovery named rational categories remain separate, as does reused T187 setup; no native scalar/runtime-equivalent price. The original decoder and failed artifacts remain unchanged. Independent saved audit pending; mathematical result scope is fixed zero-anchor/global-envelope conditional bounds only.
