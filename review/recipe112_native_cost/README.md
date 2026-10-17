# T172 native RECIPE112 arithmetic audit

## Question
Does the actual pinned NumPy RECIPE112 have a defensible arithmetic cost consistent with its comparison to sampling, after the repaired scalar wrapper and finite-panel parity audit?

## Method
Claim940aaac0 was pushed before this folder was created. T113 and T115 remain separate: no training-axis or5,120-case panel rerun. Static source is est75.py SHA426020c67a52678e5cccb20c552837fe2d233c3fe5e13557916034a3c46877d3, frozen K=(99,99), margin_all,eps.1,cliptol1e-12. `inventory.py/json` records all reachable-function operators/calls and prior T146/T153/T128 tracer hashes. Read-only prior tracer reuse must be checked against this source's actual operations rather than transferring cost multipliers.

Freeze a source-level explicit numeric convention: each scalar addition, subtraction, multiplication, division, abs, negation, comparison and elementary-function call costs1; dense dot/matrix multiply costs n*k multiplications plus n*(k-1) additions. NumPy norm costs n squares+(n-1) adds+sqrt. Cross product3D costs6 multiplications+3 subtractions. NumPy sum/mean/min/max charge reduction operations per actual output size, plus mean division. Boolean reductions are separately identified, and actual source comparison expressions are retained even when involving zeros. Model-free constant geometry is a declared reusable dependency, not free model-dependent advice. Source initialization and per-model operations are separated. Integer/index/control, Python dispatch, allocation/copy/memory and elementary-function internals are tracked or named separately; this is not hardware instruction accounting. Matrix numeric prices do not silently equate arithmetic to walltime.

Instrument an isolated AST copy, preserving evaluation order, short circuit, ndarray receivers, augmented assignment semantics, source values, legacy counters and STATS. Exclude only historical `ops.add(...)` arguments and STATS bookkeeping from estimator work. Record unsupported calls rather than silently ignoring them. No estimator optimization, pruning or tie correction. Comparing this native trace to scalar counts does not establish a universal relation.

Frozen analytic controls: exact3D basis dot/cross/norm; dense2x3 times3x2; scalar/array numeric comparisons including zero equality; computed ndarray max/sum/mean; generator min/max and short circuit; augmented assignment identity; then four H1T3 constant-first-label models from T113 (zero, zero-readout, identical-readout, dominant-first readout), plus H2T3 rational mixed-gate model. Require bitwise baseline/traced estimates, identical legacy counters and STATS; retain original tie failures. Independent T55 source/control gate before only the original H4seed1 noise89copy2 and H8seed2 noise89copy3 native traces. No trained corpus expansion unless a subsequent frozen task scope justifies it.

Two alternatives: (1) native dense/repeated arithmetic materially exceeds the scalar count despite output parity; direct source trace distinguishes. (2) accounting wrapper itself changes branches or costs; analytic operator controls and full output/counter/STATS parity distinguish. (3) a false cost increase comes from counting bookkeeping or nonexecuted branches; site ledger and independent source audit distinguish.

## Status
Static inventory complete; tracer/control implementation next. No native numerical cost conclusion yet.

## Results
The actual `_resolve_margin2.margin` evaluates `P @ A3f.T` twice and uses dense matrices for both; scalar rec112 instead computes each point separately and skips inactive terms. Source `_est75` also constructs full HxH products after masks. These source differences motivate direct accounting and prevent cost transfer. The repeated expression is preserved. Original known positive-volume tie defects are not repaired in this task.

Reproduce inventory: `/workspace/clean-neurons/.venv/bin/python review/recipe112_native_cost/inventory.py`.

## Compute accounting
Inventory parses source only, zero model evaluations. All future numerical checks one thread,2GB, review lock<=20CPU or primary<=55CPU; source/hash loading and setup tracked separately. Prior research dependencies remain explicit.

## Claims
[inferred] Direct unchanged-source accounting can close gaps invisible to the scalar Num counter. [measured] Static source contains dense/repeated operations not identically represented by the sparse scalar implementation. No cost upper bound, sampling win or ARC solution claimed.

## Alternatives not ruled out
A different efficient implementation, shared setup, compensated arithmetic, source simplification and a proof covering uninstrumented operations. Native performance may differ from arithmetic units. The recipe family remains open irrespective of this finite audit.

## Failure modes
AST evaluation-order changes; omitted NumPy methods or reductions; integer-versus-numeric category confusion; double-counting returned arrays; generator exhaustion; scalar/array coercion; source setup conflated with per-model work; floating branch changes; treating known degenerate tie defects as new trained failures.
