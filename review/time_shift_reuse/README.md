# T190 exact time-shift segment reuse

## Question
What can shared recurrent weights actually reuse across time, and which
constraints still prevent reusing whole correctness subtrees?

## Method
Claim25723b1d precedes implementation. For each fixed gate word, construct
exact affine maps and preactivation planes in incoming hidden state plus
local segment inputs. Absolute time enters only through coordinate insertion.
First five H2 synthetic models, every word of lengths1..3, exact direct
masked/ReLU controls, adjacent-segment composition and shifted coordinates.
No trained capture or geometry/mass enumeration in this phase.

## Status
Protocol/theory preparation. No numerical controls executed.

## Results
None yet. PROOF.md separates the algebraic reuse theorem from geometry,
remaining-depth, readout, truth-rank and integration obligations.

## Compute accounting
One thread2GB, review19/20CPU. Exact template arithmetic counted separately;
verification, conversion, hashing/control/storage and big-integer costs are
not free and remain separately recorded or unpriced. No Monte Carlo inputs,
trained model calls, quadrature or speed/cost advantage inferred.

## Claims
[inferred, independent review pending] Fixed gate-word templates are invariant
under time-coordinate insertion for identical recurrent weights.
[speculative] Their reuse may reduce a meaningful portion of estimator cost.

## Alternatives not ruled out
Weight sharing might help only arithmetic, while distinct cone/rank domains
still require all geometry: synthetic incoming-state/rank counterexamples
distinguish these. Apparent repeated words might arise from duplicate initial
charts: any later history census must deduplicate histories first. A template
cache can cost more than incremental construction; no speed claim from hits.
Broader affine symmetries/domain-restricted reuse are not excluded by misses.

## Failure modes
Incorrect gate-zero ownership, omitted incoming-state variables, mixed weights,
coordinate/rank relabeling errors, infeasible word counting, duplicate histories,
unpriced exact/cache setup, or equating conditional map equality with probability
or correctness equality. All counterexamples and failed attempts are preserved.
