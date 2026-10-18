# Sufficient prefix-domain future equivalence

## Question
Can different global future-margin functions be certified equal on both represented prefix cones?

## Method
Exact generator signs simplify ReLU terms on each positive hull; the reviewed canonicalizer checks a zero vector difference. PROOF.md states sufficient semantics. primitive.py and controls.py freeze16 synthetic pairs plus4 guards.

## Status
Fixed20 controls complete f557b774 after published independent source gate65cc1550. Saved audit pending; no trained application.

## Results
All16 pairs/4guards pass:12 certificates,4 retained unknowns,312 synthetic direct evaluations; .043989451 CPU. Unknown lower-dimensional linear equality is intentional. summary.json records category counts, including unused projective normalization actually performed by the canonicalizer. Fixed command (refuses existing outputs): `ulimit -v 2097152; ulimit -t 20; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/future_equivalence/domain_sufficient/controls.py`.

## Compute accounting
Explicit rational restriction and canonical arithmetic retained; observed bits and conservative sign-comparison allowance separate. Validation/GCD/decodes/control/diagnostic evaluations unpriced. No estimator or full-cost claim.

## Claims
[inferred] A zero simplified difference on both positive hulls proves equality there. [measured, saved audit pending] Fixed controls pass as above; no trained equality or performance result.

## Alternatives not ruled out
Label-only, projective or other domain equalities; unknown results cannot exclude them. A lower-dimensional linear equality is intentionally left unknown by this algorithm.

## Failure modes
Wrong ray/history binding, incomplete geometry, convex-union costs and finite-precision source arithmetic are separate. No hull-coverage, mass or sampling comparison follows from this certificate alone.
