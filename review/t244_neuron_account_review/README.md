# Independent T244 neuron-account contract review

## Question
Does the written T241 prerequisite separate encoding, computational interventions,
network fidelity and task performance, while preserving the limits of inherited
evidence?

## Method
Delegation f6544bcd, frozen author19549761. Read the contract, source inventory,
directive and relevant sections of all ten source/text payloads. Independently
check the algebra and source semantics; bind bytes to Git objects with the
metadata-only `check_provenance.py`. No author code is imported or executed.

## Status
Written source/theorem review passes. No scientific execution authorized.
The designated-folder absence is only at the pinned shared snapshot; this is
not a review of unpublished T241 code or new results.

## Results
The five label outcomes form a disjoint exhaustive partition: both correct,
erased error, introduced error, both wrong/same label, both wrong/different
labels. Agreement comprises the first and fourth; modified-minus-original
accuracy is erased minus introduced probability. Pointwise absolute correctness
change is at most disagreement, yielding the stated inequality after expectation.
Hence .99 agreement does not imply the separate .001 accuracy tolerance.

Simultaneously valid intervals for erased and introduced error probabilities
give [L_E-U_I,U_E-L_I] for accuracy change. The union bound requires no
independence between those two events. The author's conditional-on-fit iid
premises, donor/sequence dependence, multiplicity, selection and equivalence
qualifications are necessary; no numerical interval is computed here.

Source common.py includes the zero initial state and T+1 state indices; it
loads a model at import and was not imported. The mode helper restricts to the
listed 13-coordinate principal submatrix, returns eigvectors and their inverse,
and uses h @ Vinv.T. It does not supply an orthogonal basis or intervention
protocol. The state and separable helpers define retrospective final-input
features and a masked fit, not an enforced heldout protocol. That does not
demonstrate historical leakage. FINDINGS itself limits gate-set coverage and
withdraws the unqualified nonorthogonal n3/n13 attribution.

For a real basis V and coordinate residual a, ||Va||^2=a^T V^T V a; cross terms
generally remain. Complex modes require the conjugate-transpose expression and
consistent conjugate pairs for real state reconstruction. Repeated/defective
or ill-conditioned decompositions need declared handling, rather than assuming
the helper always supplies a valid inverse. This is a domain qualification of
the written contract, not a new failure observed on the target.

The intervention, donor-law and rollout distinctions are sound. Patching a
preactivation, a post-ReLU state, a terminal readout input, or a teacher-forced
trajectory defines different programs. Single-component effects do not add to
cumulative effects through recurrent gates. A well-defined off-distribution
intervention has an effect under its specified law but does not by itself
identify a neuron's unique natural semantic role. Gate-conditioned fits can
be descriptive oracle decompositions without constituting standalone predictors.

Inherited T201/T224 reports support only their stated fixed/reused-panel facts:
task accuracy can exceed network agreement, and reduced hidden error need not
improve task accuracy. This review does not repeat their array audits or extend
them to T241. Separate case denominators and overlaps, train/selection/test
roles and unknown future protocol details remain explicit.

## Compute accounting
Text reads, Git metadata and hashing only. No model/checkpoint/array decode,
forward, eigendecomposition, ablation, replacement, fit, RNG, MC or mass calls.
No sample-size, cost or runtime claim for the future owner study.

## Claims
[inferred] Written identities and protocol distinctions pass under explicit
premises. [measured] Ten source/text payloads match pinned Git bytes, lengths,
hashes and blob IDs. No new causal role, heldout score, confidence statement,
architecture ceiling, cheap reconstruction or ARC solution is established.

## Alternatives not ruled out
Frozen replacements may generalize; richer features, coordinated interventions
or other representations may be needed. Redundancy and distribution shift can
explain apparently small or large effects. No family is rejected.

## Failure modes
Confusing task accuracy with fidelity; treating mode coordinates as orthogonal;
oracle gates or original future states hidden inside a replacement; selected
test results presented as confirmatory; overlapping cases or individual effects
added as if disjoint. The document does not allege these errors in unseen code.
