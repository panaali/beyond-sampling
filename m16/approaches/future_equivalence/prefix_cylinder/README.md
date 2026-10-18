# Prefix-cylinder future margins

## Question
Does restricting to a full-dimensional prefix domain create additional exact one-step margin equalities when the next scalar input is still unrestricted?

## Method
PROOF.md isolates each surviving ReLU hyperplane inside the open prefix cylinder. A nonzero fresh-input coefficient makes each hyperplane cross every interior vertical line. Local equality forces all gradient jumps and the linear remainder to vanish globally. Reviewed weight fingerprints may then force identical state maps.

## Status
Proof-only claimd0e0cd94, written proof complete, independent h4 review pending. No numerical phase or stored-domain applicability claim.

## Results
[inferred, review pending] Exact margin equality on a cylinder is global equality under the stated hypotheses. Gates-only prefix histories are cylinders; stored truth/rank-chart cones need not be. Four exact counterexamples separate missing hypotheses and label equality.

## Compute accounting
No model/reference/geometry/numerical/rank/mass calls. Inherited predicates and prior finite-domain evidence retain their own costs. No algorithm runtime or cost saving claimed.

## Claims
[inferred] The theorem and fixed-weight consequence are conditional exact mathematical statements. [speculative] They can guide which domain-merging variants deserve further work. No claim that the ARC challenge is solved or that all mergers fail.

## Alternatives not ruled out
Final-input-constrained domains, zero input weights, lower-dimensional histories, labels, projective or approximate equivalence, reset states, and reduced domain representation costs remain separate. Initial truth charts are an explicit applicability limitation.

## Failure modes
Confusing a full-dimensional finite hull with a full cylinder, dropping first-index ties, or treating exact margin inequality as label disagreement would overstate the proof. Numerical geometry/feasibility and allpairs are not established.
