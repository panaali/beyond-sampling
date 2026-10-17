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
