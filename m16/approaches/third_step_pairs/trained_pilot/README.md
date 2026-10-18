# Five trained pair-moment pilot

## Question
Does the reviewed cone/pair method remain numerically valid on the already
pinned H16T4seed3 geometry, and what does a small fixed application cost?

## Method
Reuse T175 exact34 sectors/maps and means. Fixed zero-based pairs(0,0),(1,1),
(15,15),(0,1),(0,15),170 sector contributions. Three diagonals allow both
selected2x2 covariance tests; this choice is not based on observed errors.
Unchanged80/120-digit candidate and80-digit cylindrical reference every cell.
Compare selected entries only with existing8-block sample and three saved
approximation outputs, using explicit step3 lookup. No new forwards or samples.

## Status
Completed once after independent source gate. All170 cells pass; independent geometry2557c3d4 and aggregate19610915 audits pass.

## Results
All170 cells pass (145 pointed,25 measure-zero), maximum independent reference discrepancy4.238e-79. Both selected covariance minors positive. Candidate0.591CPU, reference6.926CPU/76,612 evaluations; total process8.572CPU. See summary.json and HANDOFF.md. Reproduction: `python run.py` in a fresh isolated output copy after preserving pinned sources; original runner refuses existing results. This is not full136-pair coverage or a full covariance PSD test.

## Compute accounting
Exact geometry/MP candidate and reference CPU percell, counts/bitwidth and all
validation/serialization process CPU retained. Imported historical sector/mean
setup remains a deployment prerequisite; no cached work is declared free.
Complete comparison/GCD/native/control/memory/FP cost/certification remains open.

## Claims
[inferred] The real Gaussian second-moment identity applies to covered cones.
[measured] Five trained pairs pass the frozen finite precision/reference checks.
[speculative] Full trained pair coverage may be feasible after this cost-first
pilot; it is not authorized here. T11595 remains blocked pending that coverage.

## Alternatives not ruled out
Shared sector errors are constrained by T175 exact map review; numerical moment
errors are tested with independent cylindrical integrals. Saved sample mismatch
can arise from sampling/ordinary floating recurrence;8-block uncertainty is
only descriptive. No later Gaussian closure or final accuracy implication.

## Failure modes
Unsupported rank<3, cancellation/precision, resource cap or covariance minor
failures remain saved. One110CPU attempt, one thread2GB primarylock; no automatic
resume, changed precision, extra pairs/models or repeated source references.
