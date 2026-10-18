# SRB per-call draw diagnostics

## Question
Can the actual completed candidate count be exposed without changing the
estimator result, its reported charge or its RNG-call order?

## Method
The local `adapter.py` preserves the pinned numerical AST and wraps only call
boundaries. `DESIGN.md` specifies separate requested/completed counts, quotas,
kept rows, original source drawn value and reported cost. Immutable snapshots
reset per call while preserving the estimator's original precompute cache.

## Status
Source/proof/control preparation only, claimdbb5a068. No functions or controls
have been called. Independent h4 source reviewer is assigned; a separately
published control execution scope/resource gate is required before any run.

## Results
No new numerical results. Reviewed T204 already showed the same584 kept rows
and raw993 charge for50000 versus100000 deterministic candidates; T214 separately
derived a policy-conditional mandatory-cost gap. This task makes observation
reviewable without changing either result or adopting a new cost policy.

## Compute accounting
Zero estimator/Q/model/random calls so far. Metadata construction, scalar
conversion, wrappers, hashing and serialization are additional diagnostic
overhead. Unchanged returned cost is preserved for parity, not endorsed as full
cost. Event counts are observations, not Gaussian-law or native-FLOP evidence.

## Claims
[inferred] Under the fixed ndarray RNG API, source call order and array identity
are preserved by forwarding wrappers and unmodified numerical AST.
[speculative] Bounded controls can validate this implementation for the frozen
fixtures; they cannot establish transparency to arbitrary user objects.

## Alternatives not ruled out
An exception may occur after an RNG internally advances state but before it
returns; report only completed batches and keep internal work unknown. A stale
per-call counter or mutable snapshot could misattribute later work; lifecycle
controls must test two calls and retained prior snapshots. Different allocations,
CV modes, concurrency and arbitrary stream implementations are outside scope.

## Failure modes
Source/API drift, observer failure, unsupported nested calls, cache corruption,
mutation of prior snapshots, extra RNG requests or changed outputs must remain
visible. No failed control can be dropped or treated as a pricing-policy result.
