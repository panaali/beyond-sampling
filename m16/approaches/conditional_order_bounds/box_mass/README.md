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
