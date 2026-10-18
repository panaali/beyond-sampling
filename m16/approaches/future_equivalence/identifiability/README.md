# Exact weight-fingerprint identifiability

## Question
Can the stronger global one-step margin-equivalence test ever join genuinely different state maps under fixed weights?

## Method
PROOF.md gives sufficient exact conditions. primitive.py computes rational RREF and all hidden-unit fingerprint subset sums. Sixteen frozen synthetic controls include collisions, zero input, singular recurrence and malformed shapes.

## Status
Claim8dc60612 pushed before implementation. Theorem checked independently by T55; recover_910 source/control gate passes4ab588b5, pinned6caca2f9. Controls98b661f7 independently reviewed9a2f2bb9; trained6d305b14 independently reviewed03c0e6a3. Bounded phase complete. Author/reviewer artifacts and cited history are now shared (export64511281 via68d6232d); the former publication blocker was resolved by explicit user authorization.

## Results
All16 frozen controls pass:7 satisfy sufficient conditions,5 fail a predicate and remain inconclusive,4 malformed/rational inputs rejected. CPU timer0 is below resolution, not free work. The later frozen trained phase used the completed independent controls gate. The finite fixtures do not exercise row swaps; exact source logic was reviewed, not all possible branches empirically exhausted.

The three stored models H4T3seed0, H4T4seed0 and H8T4seed0 all satisfy the sufficient conditions, with ranks4/4/8 and16/16/256 distinct subset sums. The independent review reconstructed exact determinants, all288 subsets and80 nonzero elimination minors; it did not import or rerun the author predicate. This supports the theorem for these exact stored weights only.

Historical commands, in the owned folder: `python controls.py` and `python trained.py`, under the frozen external20CPU/2GB review-lock cap. Outputs refuse overwrite; do not rerun completed evidence.

## Compute accounting
Exact rational arithmetic categories/observed reduced bit widths; comparison/hash/sort/GCD/temporary integer work and runtime remain unpriced. Every model-dependent predicate computation is charged research/preprocessing, not free advice. Trained phase:172 rational weight decodes, named arithmetic totals162/185/1781, processCPU0.027990574 seconds. Independent saved auditCPU0.031322382 seconds. These are not complete native-cost or estimator-MSE claims.

## Claims
[inferred] Under all sufficient conditions, global margin or positive-projective equivalence forces identical state maps. This does not establish label identifiability or a merge-cost lower bound.

## Alternatives not ruled out
Prefix-domain and label-only equivalence, identical-state reset reuse, approximate merging, other horizons and alternative state descriptions. Failed sufficient conditions leave the question open.

## Failure modes
Missing zero input weights, lost sign residues, zero/colliding subset sums, singular recurrence, or confusing all-input real-arithmetic functions with feasible source execution.
