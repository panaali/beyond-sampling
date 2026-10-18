# T221 observational contract

Source/proof/control preparation claimdbb5a068. This proposal edits no shared
estimator and changes no charge policy. Freeze the already-reviewed T204
cv.py/rb9.py, allocation/precompute, four composition fixtures and saved baseline
outputs before any execution. `adapter.py` extracts the exact same selected
original function/constant AST nodes; it does not change arithmetic, selection,
allocation, precompute cache, original Q or return expressions. Source hashes
and AST identity gate the restricted strat=True,cv=None API.

An instance owns one extracted source namespace/cache and one live observation
at a time. The wrapper calls the original estimator exactly once. Its RNG proxy
forwards exactly each source standard_normal((batch,9)) invocation to the caller
and returns the identical array object. The draw wrapper returns the identical
groups/drawn tuple. Only metadata are added; no arrays are retained by a snapshot.
Each completed call returns the exact original (estimate,cost) object. Exceptions
propagate, and finally freezes immutable scalar/tuple metadata then clears live
state. Previous snapshots remain valid after later calls; original _pc remains
cached across calls. Nested observations reject before altering outer state.

Report requested candidate count separately from completed returned candidate
count. If RNG raises, its requested batch is visible but its internal partial
generation/state advancement is unknown. If classification/Q raises after a
returned batch, completed count remains visible even when source drawn/kept or
reported charge are unavailable. None means unobserved, never zero by default.
On success source_drawn and completed candidates should coincide for the fixed
standard ndarray API; quota/kept/charge are preserved separate quantities.

Observational equivalence is conditional on ordinary successful metadata
operations and the explicitly scoped RNG ABI. The wrapper is not transparent to
introspective RNGs that inspect call stacks/identity, concurrent calls, memory
exhaustion, or arbitrary malformed return objects. Additional metadata work is
research/runtime overhead, not claimed zero-cost or included in the unchanged
reported estimator charge. No rule5 interpretation, pricing repair, native cost,
historical candidate count, Gaussian rejection rate or MSE claim is made.

Planned fixed controls, still unexecuted:

1. Reuse the four original deterministic composition cases and saved baseline
   outputs/requests; one new observed call per case, no original Q/estimator
   baseline rerun. Exercise one instance across calls to verify cache/snapshot
   lifecycle. These streams are API fixtures, not Gaussian-law samples.
2. Freeze separate synthetic exception/lifecycle cases for failure before any
   completed RNG return, after one completed batch, and after draw completion
   but before Q returns. No inference about RNG internal partial work.
3. Exact scalar snapshot immutability and nested-call protection; preserve
   outer state and every failure. Any synthetic Q replacement is labeled and
   not used for the four original-Q comparisons.

A later frozen protocol and independent source/resource gate precede all calls.
Controls use review lock,one thread,20CPU/2GiB total, durable attempts/results and
no automatic retries. Independent saved verification precedes any handoff.
There is no actual-model/Gaussian/native campaign in this task.
