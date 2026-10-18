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
Bounded witness and actual-prefix extension complete. Independent T55 exact
suffix review868c69b and actual-prefix extension review145d6db pass.

## Results
Results below. This is a complement discriminator, not a full accuracy estimator.
Reproduce from this folder with the project interpreter: `python controls.py`,
`python pilot.py`, `python prefix_support.py`; one thread and review lock.

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
Actual-prefix open neighborhoods now realize both outputs, but the constructed
second event is astronomically rare. Conditioning on prior hidden state,
using a longer suffix, marginalizing
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

## Actual-prefix support qualification, frozen before implementation
TASKS scope29a5e8b pushed. The two chosen initial states can be realized by
six-input prefixes A=(1/4,0,0,0,0,0) and
B=(1/4,0,0,0,0,-256/alpha), alpha=-Wi[0]. The first positive input puts hidden
on axis2; the next zero resets it because the second Wh column is negative.
The final large negative input of B produces(256,0). Both complete center
sequences have the same unique second-largest index0, because the suffix
contains +1 and three -1 values while the first prefix input is1/4.

A pointwise center is not a positive-probability prefix event. Define exact
M=||Wh||infinity, B0=||Wi||infinity, L6=B0*sum(M^k,k=0..5).
A prefix L-infinity perturbation epsilon changes its final hidden state by
at most L6*epsilon. A suffix gate preactivation at step t changes by at most
M^(t+1)*L6*epsilon; an output margin changes by at most
||Wo[j]-Wo[k]||1*M^4*L6*epsilon. Choose epsilon<=1/16 and every retained
center slack divided by four times its respective bound coefficient. The
existing suffix box uses at most half each slack, and the prefix error at
most one quarter, leaving a strict margin. Every gate constraint must be
present; an ignored identically-zero readout margin must remain identically
zero on the entire final active support, not merely at fixed initial h.

Independent direct Fraction recurrence checks all1024 ten-input box vertices
for each of the two boxes and verifies unique target index0. T55 independently
checks the Lipschitz/slack argument before measurement. Exact lower Gaussian
mass is stored in factored rational form to avoid enormous integer strings;
80-digit stable-tail CDF products give a numerical log probability separately.
The B prefix includes a Gaussian input near-403, so its probability is
astronomically small. This phase establishes support/identifiability, not a
practically important remaining error or a competitive full estimator.

## Actual-prefix support result
`prefix_support.py/json` constructs both exact rational prefix boxes and
checks all25 suffix gate/readout inequalities with the additional prefix
perturbation. All9 omitted readout comparisons belong to the zero-final-hidden
case; their coefficients vanish on its entire active support. They therefore
remain exact ties after prefix perturbation, with first-index label0 retained.
The common prefix radius is approximately1.51835e-7. Each retained inequality
has at least one quarter of its original exact slack throughout the full box.

All2048 direct Fraction ten-input vertex recurrences pass, with output0 for A,
output6 for B, and unique second-largest input index0 for both. The whole-box
claim follows from the exact bounds, not interpolation between tested vertices.
CPU .346812 including rational construction, vertices and80-digit probabilities.

The numerical ten-input box masses are4.55853872891e-48 and
6.29544730648e-35396. Separate rigorous lower bounds have factored form
`(2*epsilon)^6*(2*rho)^4/3^P`, where P=20 and81411. Their log10 values are
approximately-52.01054 and-38885.38658. Factored exact positivity does not
depend on the numerical CDF computations. The far-negative interval uses
erfc differences to avoid cancellation.

[inferred, independently checked; DRI classification pending] Even under the actual Gaussian input
law, exact prediction/correctness cannot be determined almost surely from
these four suffix inputs alone: the same positive-mass suffix box admits
positive-mass prefix boxes with different output and correctness. This does
not bound the practically important mass of the unresolved complement, and
does not exclude a competitive approximate suffix-based estimator.

Independent extension review145d6db rebuilds the two center states, all25
quarter-slack bounds, complete gate coverage, robust omitted readout ties,
unique rank index0 and exact Gaussian denominator powers20/81411. No broad
family failure or practically important error floor follows from this proof.
