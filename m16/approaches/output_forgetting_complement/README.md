# T105: exact complement ambiguity under four-input output forgetting

## Question
Does the remaining complement necessarily contain positive-probability suffix
inputs whose output still depends on the previous hidden state, under the
same arbitrary-nonnegative-state assumption as the existing certificates?

## Method
Fix the public H2T10 seed0 weights as their exact binary rational values.
Test four-input suffix words in{-1,+1}^4 and initial states0,(s,0),(0,s) for
s=1,16,256 in a fixed lexicographic order. For each word/state propagate exact
Fraction affine functions of its four suffix inputs, retaining every gate
inequality and first-index readout margin. A candidate pair of states must
have different outputs and strict slack on every nonconstant active branch
constraint at the shared suffix center. Identically zero gates use the
inactive branch; positive-volume readout ties belong to the earliest index.

For each retained affine inequality a*x+b>0 with center slack m>0, the common
box radius rho<=m/(2*sum|a|) preserves its sign. Cap rho at1/4 and take the
minimum across both state histories. This yields an exact rational open box
on which the two outputs differ, not a pointwise floating counterexample.
Only the first certified pair in fixed order is reported as the witness;
we do not optimize model accuracy or choose a favorable measured probability.
If no witness occurs, one predeclared stronger finite control uses words in
{-2,-1,1,2}^4 with the same states. If that also fails, conclude only that this
finite witness search is inconclusive, not that the output forgets all states.

The product Gaussian box probability is numerically evaluated at80 digits.
An independent exact positive lower bound uses pi<4,e<3: if every coordinate
has |x|<=R and K>=R^2/2 is an integer, phi(x)>3^(-(K+1)), hence box mass>
(2rho/3^(K+1))^4. This rational lower bound is a true positive-mass statement;
the high-precision CDF difference alone is not an outward certificate.

## Alternatives and falsifiers
1. The complement requires state information: an exact open-box witness
   disproves complete four-suffix uniform-state classification.
2. The chosen states/words miss ambiguity: finite no-witness result is
   inconclusive; actual-prefix-reachable states remain a separate question.
3. Floating branch errors create a false witness: different direct Fraction
   recurrence at all16 box vertices and rational coefficient/slack checks
   must reproduce the two labels independently of the affine builder.
4. The witness changes only tied measure-zero cases: require a full box and
   strict gate/readout slack unless the affine constraint is identically zero.

Toy controls: memoryless Wh0 cannot distinguish prior states; constant and
duplicate outputs cannot do so; an identity-memory two-state toy admits a
box. Exact first-index ownership and rescaling are checked. Independent
review precedes any scoped impossibility statement. No previous eight-event
integrals are repeated, and no new uniform-output event family is added.

## Status
Claim63660ab pushed before implementation. Existing T105 author confirms
no overlap; its eight-event phase is complete and T121 is its current task.

## Results
Pending. This is a complement discriminator, not a full accuracy estimator.

## Compute accounting
One thread,2GB, <=20CPU/review lock for the bounded finite search and controls.
Record exact recurrence calls, affine operations, candidate words/states,
certificate constraints, validation vertices and full CPU. Fraction and
high-precision special-function cost remain explicit/opaque, not cheap FLOPs.
No random model inputs, training, fitted parameters or shared-source edits.

## Claims
[inferred] A certified positive-mass ambiguity box rules out complete output
forgetting based only on four suffix inputs for every nonnegative initial h.
[speculative] The complement has practically important state dependence.

## Alternatives not ruled out
Both witness states need not be reachable under the actual six-input prefix
law. Conditioning on prior hidden state, using a longer suffix, marginalizing
its distribution or integrating the full network remains possible. No broad
negative about mechanistic estimation, the original model or the challenge.

## Failure modes
Measure-zero gates/ties, dropped affine offsets, wrong Gaussian measure,
claiming arbitrary-state witnesses are actual-prefix probabilities, and
confusing one positive-mass obstruction with a maximal complement description.

## Frozen witness result
Toy memoryless, constant and duplicate cases have no witness in the16-word
control panel; the identity-memory toy does and passes exact vertex checks.
These controls cost .126CPU. The trained search finds its first certificate
on the third word, suffix(-1,-1,+1,-1), after21 exact histories and45 state-pair
checks (.0262CPU including validation and high-precision probability).

Initial hidden(0,0) produces label0; initial hidden(256,0) produces label6
(zero-based). A common open L-infinity suffix box has exact radius
928873407584354211505/12871199735412830144098, approximately .07216680859.
All32 independently executed state/vertex recurrences preserve the labels;
every retained affine margin is at least half its center slack. Exact hidden
rescaling2^-30 preserves labels and radius and passes32 more direct checks.

The Gaussian suffix-box mass is approximately1.48771758938e-6 by80-digit
CDF evaluation. The separate exact rational lower bound exceeds6.61454e-8,
using phi>1/9 on this box. Therefore, subject to independent review, any
four-suffix classifier required to be correct for every nonnegative initial
hidden state must leave at least this positive mass unclassified. It does
not establish a large practical coverage ceiling or that either witness
state occurs with meaningful probability under the six-input prefix law.
The current result is a scoped obstruction to universal four-input output
forgetting, not a negative about state-conditioned integration or full accuracy.
