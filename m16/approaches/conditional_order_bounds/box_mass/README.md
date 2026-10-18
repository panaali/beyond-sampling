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
Source frozen for independent review; no controls or trained contraction run yet.

## Results
Pending. Every sufficient-test failure will remain unresolved.

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
