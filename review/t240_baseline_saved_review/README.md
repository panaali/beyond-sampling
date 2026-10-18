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
The sole diagnostic saved audit passed: all 81 archive pairs, 20,736 rows, 82,944 gap cases, 83 progress events, integer sums, original counters and source/model bindings. Review CPU 2.919142395 seconds, peak RSS 35,812 KiB. There was no numerical reviewer failure or rerun.

| Actual physical gap | Preserved original report key | Selected correct mass (rounded) | Event probability enclosure (rounded) |
|---|---|---|---|
| 0 | 0 | 0.0682131361 | [1, 1] |
| 1/8 | 1/4 | 0.0592013986 | [0.0994106276, 1] |
| 1/4 | 1/2 | 0.0592013986 | [0.0994106276, 1] |
| 1/2 | 1 | 0.0592013986 | [0.0994106276, 0.9588655668] |

Exact rational lower/upper values are retained in `results.json`; the table is only a rounded display. Selected wrong mass is zero throughout. Physical gap 1 remains untested. The original intended-gap protocol remains failed despite successful diagnostic arithmetic; no broad method failure follows.

Execution gate 7fad5121 was published at 5941777c after source review 216d37f1/shared52abd9c5. Exact command:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'set -euo pipefail; set -o noclobber; flock /workspace/work/codex-review.lock bash -c "set -euo pipefail; ulimit -St 19; ulimit -Ht 20; ulimit -Sv 2097152; ulimit -Hv 2097152; timeout 25 /workspace/clean-neurons/.venv/bin/python review/t240_baseline_saved_review/check.py" > review/t240_baseline_saved_review/run1.log 2>&1'
```

## Compute accounting
One saved-only review completed under 19 soft / 20 hard CPU seconds, 2 GiB address space and 25 seconds wall time after acquiring the shared review lock. Hashing, decompression and checking are reviewer work. Author recorded CPU 2.2023471100000003 seconds is preserved, not independently timed. No affine recurrence, network/model/checkpoint, Gaussian CDF, RNG, integration or fit calls occurred. All prior source margins and probability bounds remain reviewed premises, not recalculated quantities.

## Claims
- [inferred] A saved contraction can check the baseline bookkeeping independently without repeating the scientific certificate construction.
- [inferred] Inherited network uncertainty and whole-box boundary loss are distinct limitations.

## Alternatives not ruled out
Partial-box event integration could tighten the boundary envelope; improved network bounds could tighten the inherited unknown fraction. Neither is implemented or tested here. Actual event mass and complete estimator cost remain separate questions.

## Failure modes
Misbound old archives, incorrect first-index ownership, duplicate or missing rows, assigning whole-box mass to a partial intersection, and inconsistent integer denominators can invalidate the contraction. Every failed review attempt will be preserved. This audit does not independently reconstruct all old affine enclosures or certify runtime cost.
