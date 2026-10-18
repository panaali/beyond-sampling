# Independent T240 written certificate contract review

## Question
Are the proposed separated-gap domain, probability accounting and correctness
certificates valid, without mistaking a prefix or filtered old certificate for
new full-target coverage?

## Method
Delegated ba46c4f3. Review author3776d1b3 and correction dc183c14, the pinned
T189 theory/source/handoff and T219 source documents. `check_provenance.py`
checks only text hashes, Git bytes and publication ancestry. No data archives
or model weights are decoded, and no author code is called.

## Status
Pass after the preserved numerical-ceiling wording correction. This is a
written proof/protocol gate, not execution clearance for any proposed phase.

## Results
For each ordered distinct top/second label pair i,j, condition on x_j=t.
The top coordinate is at least t+g and every other coordinate at most t-g.
Independence gives the product [1-Phi(t+g)]Phi(t-g)^(T-2); the T(T-1) label
pairs are disjoint almost surely. Multiplying by phi(t) and integrating proves
the displayed event probability. At g=0 the beta integral equals 1. This is
not a numerical quadrature certificate. Positive absolute g produces affine
polyhedra rather than homogeneous cones.

The box containment predicate minimizes each required coordinate difference
over the Cartesian box. The intersection predicate is necessary by the bounds
on x_j and sufficient by choosing other coordinates at their lower endpoints
and x_i=max(l_i,x_j+g). Endpoint-only intersections may be kept in an outer
cover. Each box is counted once; only a unique almost-sure truth label permits
the first inner construction. The outside cube remains unresolved.

The inherited all-rival margin inequalities correctly preserve first-index
readout ties. Dependence between affine inputs and retained interval residuals
need not be probabilistically independent for an enclosure. The earlier 1/m
rank-ownership law cannot simply be conditioned on the new gap event; whole-box
gap ownership avoids that unsupported transfer.

Let p be event mass, C,W disjoint certified-correct/wrong masses. Actual correct
mass in the event lies between C and p-W. Using p<=p_hi yields the conservative
conditional bounds c_lo/p_hi and 1-w_lo/p_hi when p>0. The unresolved mass
p-C-W lies in the stated difference interval intersected with [0,1], subject
to consistency rather than silently repairing contradictory bounds. These are
event/accuracy enclosures, not an estimator MSE or cost comparison.

Filtering preserves a subset of an existing certificate event. The original
3776d1b3 incorrectly used a displayed lower bound alongside a numerical ceiling;
dc183c14 removes that inference and requires the original upper endpoint and
provenance before a numeric ceiling is asserted. The initial issue remains in
Git and REVIEW_CORRECTIONS.md. No archived mass was read to repair the wording.

On bounded rational domain restrictions, affine extrema at a complete vertex
set or valid LP dual certificate can support stronger ranges. Sampled vertices
cannot. Retaining parent certificates under refinement gives monotone coverage
independently of whether a new local approximation is looser. Exact prefix
histories alone omit unbounded future inputs, final rank ownership and suffix
computation; they cannot count as full-target certified-correct mass.

## Compute accounting
Text, metadata, hashing and written algebra only. Zero author imports, arrays,
checkpoints, models, certificate evaluations, CDF/quadrature, RNG or geometry.
Historical partial integer ledgers and CPU are not repriced or transferred.

## Claims
[inferred] Corrected identities and prospective scope pass under their stated
independence, completeness and exact-real assumptions. [measured] Bound source
text matches published bytes. No easy-mass amount, full cost or improvement is
demonstrated. Later controls, saved filtering and stronger candidate phases each
need their separately frozen source/resource and evidence gates.

## Alternatives not ruled out
Finer or nonrectangular domains, exact conditional integration, reachable-state
constraints and other correlated enclosures may help. Low coverage in this
finite construction would not prove intrinsic error complexity or capacity.

## Failure modes
Lower bounds used as upper ceilings; center labels used for whole boxes;
absolute gaps treated as cones; inherited rank probabilities reused after
conditioning; prefix success reported as final accuracy; uncharged exact setup.
