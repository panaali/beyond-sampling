# T240 independent saved H8 baseline review

## Question
Does the frozen four-gap baseline correctly contract the previously reviewed H8T4 box certificates, without assigning probability to unmeasured partial-box intersections?

## Method
Delegation 99f930e6 reserves this folder. The planned independent checker visits all 81 original and 81 baseline archives, all 20,736 rows and gaps 0, 1/4, 1/2, 1. It will bind completed source/review gates and raw artifacts before decoding, reconstruct lexical bin ownership, inspect interval endpoint winners for first-index label semantics, and independently derive positive-gap box containment and feasible intersections. Integer mass sums, four event/selected-correct/selected-wrong/unresolved summaries, progress coverage and inherited accounting will be checked exhaustively. No author implementation is imported.

At zero gap, the original within-bin second-label law is retained. For positive gaps, only wholly contained boxes contribute selected correct/wrong mass; partial boxes stay unresolved. The audit does not evaluate their actual intersection probabilities.

## Status
Source-unit defect found before checker execution or any archive decoding; see `UNIT_FINDING.md`. The author's completed positive-gap run represents half the stated physical gaps. Raw outputs remain preserved. The initial unexecuted checker draft is retained as `check_initial_unrun.py`; no reviewer execution gate exists.

Renewed scope 0abba9ac permits a diagnostic-only saved contraction at the actual physical gaps {0, 1/8, 1/4, 1/2}. `check.py` retains the original report keys and records physical keys separately. Physical gap 1 is untested. This diagnostic cannot turn the intended campaign into a pass. The independent checker and `scope.json` are frozen for source review before any archive or baseline result decoding.

## Results
No numerical review result. A source-only cross-check proves the current positive-gap predicates use physical gaps {1/8, 1/4, 1/2}, while the frozen contract states {1/4, 1/2, 1}. This does not establish a failure of the underlying method.

## Compute accounting
Planned one saved-only review under 19 soft / 20 hard CPU seconds, 2 GiB address space and 25 seconds wall time after acquiring the shared review lock. Hashing, decompression and checking are reviewer work. No affine recurrence, network/model/checkpoint, Gaussian CDF, RNG, integration or fit calls are authorized. All prior source margins and probability bounds remain reviewed premises, not recalculated quantities.

## Claims
- [inferred] A saved contraction can check the baseline bookkeeping independently without repeating the scientific certificate construction.
- [inferred] Inherited network uncertainty and whole-box boundary loss are distinct limitations.

## Alternatives not ruled out
Partial-box event integration could tighten the boundary envelope; improved network bounds could tighten the inherited unknown fraction. Neither is implemented or tested here. Actual event mass and complete estimator cost remain separate questions.

## Failure modes
Misbound old archives, incorrect first-index ownership, duplicate or missing rows, assigning whole-box mass to a partial intersection, and inconsistent integer denominators can invalidate the contraction. Every failed review attempt will be preserved. This audit does not independently reconstruct all old affine enclosures or certify runtime cost.
