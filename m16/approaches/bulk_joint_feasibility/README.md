# T120 exact simultaneous bulk-feasibility discriminator

## Question
Can simultaneous gate, input-rank and readout constraints resolve the largest
H4T4 gate histories that conservative label-set certificates left unresolved?
Claimcfbe199 was pushed before implementation. This is a bounded discriminator,
not another full gate enumeration or a completed accuracy estimator.

## Method
Load the fixed T68 exact H4T4 artifact and prior T116 unresolved classification.
Recompute full gate Gaussian masses with unchanged positive GL20 rule, sum by
history, and freeze the ten largest unresolved histories using only these
masses (tie-break history bit string). Do not use correctness/error fractions
or model truth to choose the panel. Persist selection before feasibility runs.
Record selected mass separately from certified resolved coverage.

For each history reconstruct integer/dyadic gate planes and output maps from
weights. For each ordered top/second input pair add strict rank inequalities;
inside an open full-dimensional cone every nonzero gate plane is strict too.
A correct event requires second-input output to beat every competitor, with
first-index ownership of identically equal maps. An error event exists if any
competitor beats that target; identical earlier competitors force rejection,
while identical later competitors do not. Nonzero margins can be treated
strictly for positive Gaussian mass; lower-dimensional boundaries are excluded.

For rational A, existence of x with Ax>0 is equivalent to Ax>=1. Floating LP
can propose a witness, but accept only a Fraction/dyadic x with every original
integer inequality strictly positive. Infeasibility is accepted only with an
exact normalized nonnegative vector lambda: A^T lambda=0, sum(lambda)=1.
A floating alternative-dual solution may propose its sparse support; reconstruct
weights with exact linear algebra and verify every identity/sign independently.
No solver status, tolerance, failed rationalization or elapsed limit counts as
infeasibility. Return unknown when neither exact certificate is obtained.

Freeze analytic controls: positive orthant; opposing normals with no strict
solution; nontrivial three-plane positive dependence; thin feasible wedge;
positive rescaling invariance; zero rows; constant/duplicate outputs and proper
first-index ties. Compare direct recurrence and rank at trained primal witnesses.
Budget total240 LP calls and20CPU per phase, one thread/2GB/review lock; checkpoint
after each history/pair and preserve partials if exhausted. No unbounded rescue.
Process histories in frozen descending-mass order, rank pairs lexicographically.

## Alternatives and falsifiers
1. Conservative separate label sets hide strong joint exclusions: exact duals
may prove whole rank-conditioned regions impossible without enumerating them.
2. Large cells genuinely contain both correct and wrong open sets: exact paired
witnesses show that a whole-cell constant-correctness shortcut cannot resolve them.
3. Numerical LP difficulty masquerades as exclusion: exact witness/dual audit
and thin/rescaled controls prevent that inference; failures stay unknown.
4. Many certificates cover negligible mass: report selected gate mass, actual
whole-gate certification and uncovered mass separately; no mass from counts.

## Status
Protocol frozen. Mass-only selection next, followed by controls before trained
feasibility. Completed T110 offers rank-family normals but has no full terminal
cell cache; no duplicate engine is required for these halfspace checks.

## Results
Pending. Known model accuracy is diagnostic only, never a selection/fitting input.

## Compute accounting
Charge artifact decompression, gate mass integration, integer map reconstruction,
all primal/dual LP attempts and exact certificate verification, including failed
attempts. Existing T68 construction is a historical cold dependency. No portable
FLOP/sampling advantage follows from LP counts or this diagnostic runtime.

## Claims
[inferred] Exact open witnesses and positive-dependence certificates establish
strict feasibility/infeasibility under the checkpoint's rational real-arithmetic
model. [speculative] Joint constraints permit useful coarse bulk estimation.

## Alternatives not ruled out
Top-two-conditioned mass integration, cross-gate unions, stronger direct cost
bounds, weak-memory approximations and other model/rung cases remain open.

## Failure modes
Boundary/positive-volume confusion; first-index identical margins; scaling of
floating LP proposals; nonbasic numerical dual support; exact integer bit growth;
resource exhaustion; and mistaking selected mass or existence of two events for
the quantitative error mass inside a mixed gate cone.

## Frozen selected panel
`select.py` persisted ten histories before any feasibility solve. Their total
input mass is.4482031037438533, with.5517968962561467 outside the selected
panel. This is selected mass, not resolved accuracy coverage. Selection costs
.937495CPU including artifact/source setup; the complete gate partition is
1.0000000000000002 with conditional quadrature bound5.79e-8. Every history,
individual mass/bound and source hash is saved in selection.json. No correct
mass or per-history error rate was used by selection.

## Preserved thin-wedge solver failure and bounded rescue
The frozen thin feasible system x>0, y>10^12*x returns floating primal
infeasible after row normalization; the numerical dual proposal fails exact
balance and is correctly reported UNKNOWN, never infeasible. The control
requires a real witness, so trained runs are not started. controls_failure.json
preserves the failure. This is LP conditioning, not a mathematical obstruction.

Before changing the implementation, freeze exact positive column equilibration:
after rational row normalization, divide each variable column by its maximum
absolute coefficient (use1 for identically zero columns). Map any proposed
primal back with the inverse positive scale, then verify all original integer
inequalities. The dual nullspace is unchanged by invertible column scaling;
reconstruct against the original rational row-normalized matrix and verify
original integer balance. Rerun the same ten controls, no new threshold or
weakened exact acceptance rule. Solver unknown remains a valid unresolved state.

Exact column equilibration repairs all ten frozen generic controls with12 LP
calls in.312634CPU, including the1e12 thin wedge and its rescaling. Separate
constant/duplicate first-index controls and zero-gate OFF/equality removal pass.
A strict zero row remains infeasible in the generic solver; semantic zero gate
forms are removed by the model adapter, preserving full-dimensional equalities.
Original failure remains saved. Trained feasibility has not been used to choose
these repairs or acceptance thresholds.

## First bounded trained panel
The frozen240-LP cap stops after seven complete histories and one partial rank
pair in the eighth, in1.234235CPU including setup/checkpoints (95MB RSS). All
accepted systems carry exact verified primal or normalized nonnegative dual
certificates. Completed rank pairs:48 empty,14 uniformly wrong,2 uniformly
correct,21 mixed; one pair remains partial. These are counts, not mass coverage.

Every one of the seven fully examined gate histories has both a correct and
an incorrect open input witness, including a history whose individual rank
families are each constant-correctness. Their gate mass totals.36240678175283747.
Thus those whole gate cones cannot be assigned a single exact correctness value;
internal rank/readout splits are necessary for exact treatment. No whole-gate
uniform certificate was obtained, so resolved whole-gate mass is0. The full
selected panel remains.4482031037438533; unprocessed histories/pairs are not
called infeasible or discarded. Feasibility says nothing about the relative
masses of the two witnessed events inside a mixed cone.

The output is intentionally incomplete at its predeclared call cap; no silent
budget extension occurs. `panel.json` preserves original integer systems,
exact certificates, direct recurrence/rank checks and every partial row. Source
review has validated Gordan semantics and exact column rescaling; independent
verification of trained certificates is requested before interpreting the
negative. Future mass attribution of rank-conditioned certified regions is a
separate scope, potentially reusing existing T110/T68 decompositions. This
result does not rule out those conditioned or approximate mechanisms.
