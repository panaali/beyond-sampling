# T229 — exact positive-monomial weight symmetry obstruction

## Question
Do exact readout columns alone force identity for global-state positive
rescaling/permutation transformations that leave the network weights unchanged?

## Method
`PROOF.md` states the narrow necessary readout condition. `predicate.py` uses
exact positive-ray canonicalization and explicit2x2-minor/proportionality
certificates. `controls.py` freezes20 analytic/API cases before execution.

## Status
Claim903f13f6 published. All20 controls completed9335fac5 after independent
source gateb1bcf15e and saved review4e5eb0b9. Target source1b2dab3d is separately
frozen/published; no target predicate has run, independent source gate pending.

## Results
[measured]12 valid/8 invalid controls pass once in0.02808157CPU; see
CONTROL_RESULTS.md. [inferred] Pairwise nonzero, non-positive-proportional
readout columns suffice to rule out every nonidentity global positive-monomial
weight automorphism. This exact algebra is independent of approximate cosines.

## Compute accounting
Control phase0.02808157CPU/25,300KiB under16soft20hardCPU/2GiB. No target/archive,
forward, geometry, mass, RNG or fit calls. Exact Fraction preparation is charged as
research CPU; partial counts are not portable/native cost.

## Claims
The theorem is [inferred], pending independent review. No target conclusion,
ARC solution, broad function-equivalence exclusion or runtime benefit.

## Alternatives not ruled out
Argmax-only/common-shift invariances, reachable-domain equivalences, noninvertible
quotients, approximate transformations and all representations outside the
stated class. Readout ambiguity returns inconclusive, not a positive symmetry.

## Failure modes
Incomplete provenance, nonfinite/nonexact API inputs, omitted zero-column cases,
confusing a change of parameterization with an automorphism of fixed weights,
and overextending exact logit invariance to every output-label equivalence.
