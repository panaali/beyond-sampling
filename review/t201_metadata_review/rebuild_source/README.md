# T201 synthetic helper source gate

## Question and method
Does the frozen seven-group analytic driver test the intended helper semantics
before any real checkpoint fit? Assignmentf2a5fece expands the prior metadata
review after T55 yielded this source gate. Read code only; no author imports.

## Status and results
Source3fe1d31c passes the bounded gate in source_gate.json. The three3-feature
lookup training matrices have determinants−1,−1,+1. For each distinct position
pair, the separable training values (m2,gap)=(2,1),(2,2),(1,2) force both
coefficients separately, and all positions occur, establishing full18-column
rank in real arithmetic. Numerical recovery remains a fixed finite control.

The cross-tab control now checks all inverse ranks against independently sorted
raw rows and obtains predicted labels from those independent orders. This fixes
the earlier circular control before any execution. SSE/SST/None semantics,
strict no-resample input ties, first-index output ties,219 total solves including
deficient cases, and six scalar-Fraction recurrence fixtures are consistent.

## Compute accounting and claims
Zero numerical/source executions by reviewer. One externally capped20CPU/2GB
one-thread author process is cleared only after immutable gate publication.
This is a source gate; no trained performance, historical-fit or cost result.

## Alternatives and failure modes
Do not relax1e-10 or rcond after results; preserve all failures. Actual model
fitting, fresh random data and old-rate comparison need new scope and independent
saved control review. Finite controls do not prove universal floating correctness.
